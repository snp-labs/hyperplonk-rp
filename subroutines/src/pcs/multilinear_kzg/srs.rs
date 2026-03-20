// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Implementing Structured Reference Strings for multilinear polynomial KZG
use crate::pcs::{multilinear_kzg::util::eq_eval, prelude::PCSError, StructuredReferenceString};
use ark_ec::{pairing::Pairing, scalar_mul::BatchMulPreprocessing, AffineRepr, CurveGroup};
use ark_ff::{Field, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    end_timer, format, rand::Rng, start_timer, string::ToString, vec::Vec, One, UniformRand,
};
use std::env;
#[cfg(target_os = "macos")]
use std::mem::MaybeUninit;

/// Evaluations over {0,1}^n for G1 or G2
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct Evaluations<C: AffineRepr> {
    /// The evaluations.
    pub evals: Vec<C>,
}

/// Universal Parameter
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct MultilinearUniversalParams<E: Pairing> {
    /// prover parameters
    pub prover_param: MultilinearProverParam<E>,
    /// h^randomness: h^t1, h^t2, ..., **h^{t_nv}**
    pub h_mask: Vec<E::G2Affine>,
}

/// Prover Parameters
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct MultilinearProverParam<E: Pairing> {
    /// number of variables
    pub num_vars: usize,
    /// `pp_{0}`, `pp_{1}`, ...,pp_{nu_vars} defined
    /// by XZZPD19 where pp_{nv-0}=g and
    /// pp_{nv-i}=g^{eq((t_1,..t_i),(X_1,..X_i))}
    pub powers_of_g: Vec<Evaluations<E::G1Affine>>,
    /// generator for G1
    pub g: E::G1Affine,
    /// generator for G2
    pub h: E::G2Affine,
}

/// Verifier Parameters
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct MultilinearVerifierParam<E: Pairing> {
    /// number of variables
    pub num_vars: usize,
    /// generator of G1
    pub g: E::G1Affine,
    /// generator of G2
    pub h: E::G2Affine,
    /// h^randomness: h^t1, h^t2, ..., **h^{t_nv}**
    pub h_mask: Vec<E::G2Affine>,
}

fn extend_eq_suffix_table<F: Field>(prev: &[F], ti: F) -> Vec<F> {
    let mut next = Vec::with_capacity(prev.len() << 1);
    let at_zero = F::one() - ti;

    for &value in prev {
        next.push(value * at_zero);
        next.push(value * ti);
    }

    next
}

#[derive(Clone, Copy, Debug)]
struct SetupMemorySnapshot {
    resident_size: u64,
    phys_footprint: u64,
}

fn setup_memory_logging_enabled() -> bool {
    env::var_os("AJTAI_SETUP_MEM_LOG").is_some()
}

#[cfg(target_os = "macos")]
#[allow(deprecated)]
fn current_setup_memory_snapshot() -> Option<SetupMemorySnapshot> {
    unsafe {
        let mut basic = MaybeUninit::<libc::mach_task_basic_info>::zeroed();
        let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
        let basic_result = libc::task_info(
            libc::mach_task_self_,
            libc::MACH_TASK_BASIC_INFO,
            basic.as_mut_ptr().cast::<libc::integer_t>(),
            &mut count,
        );
        if basic_result != libc::KERN_SUCCESS {
            return None;
        }
        let basic = basic.assume_init();

        let mut usage = MaybeUninit::<libc::rusage_info_v4>::zeroed();
        let mut usage_buffer = usage.as_mut_ptr().cast::<libc::c_void>();
        let usage_result =
            libc::proc_pid_rusage(libc::getpid(), libc::RUSAGE_INFO_V4, &mut usage_buffer);
        if usage_result != 0 {
            return Some(SetupMemorySnapshot {
                resident_size: basic.resident_size,
                phys_footprint: 0,
            });
        }
        let usage = usage.assume_init();

        Some(SetupMemorySnapshot {
            resident_size: basic.resident_size,
            phys_footprint: usage.ri_phys_footprint,
        })
    }
}

#[cfg(not(target_os = "macos"))]
fn current_setup_memory_snapshot() -> Option<SetupMemorySnapshot> {
    None
}

fn format_bytes(bytes: usize) -> String {
    let units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;

    while value >= 1024.0 && unit + 1 < units.len() {
        value /= 1024.0;
        unit += 1;
    }

    if unit == 0 {
        format!("{bytes} {}", units[unit])
    } else {
        format!("{value:.2} {}", units[unit])
    }
}

fn log_srs_setup_memory<E: Pairing>(
    checkpoint: &str,
    suffix_len: Option<usize>,
    powers_of_g: &[Evaluations<E::G1Affine>],
) {
    if !setup_memory_logging_enabled() {
        return;
    }

    let total_affine_points = powers_of_g
        .iter()
        .map(|bucket| bucket.evals.len())
        .sum::<usize>();
    let retained_bytes = total_affine_points.saturating_mul(core::mem::size_of::<E::G1Affine>());
    let suffix_len = suffix_len.unwrap_or(0);
    let suffix_bytes = suffix_len.saturating_mul(core::mem::size_of::<E::ScalarField>());
    let snapshot = current_setup_memory_snapshot();

    eprintln!(
        "[mlkzg-srs-mem] checkpoint={} suffix_len={} suffix_scalar_bytes={} powers_of_g_chunks={} powers_of_g_points={} powers_of_g_bytes={} resident_size={} phys_footprint={}",
        checkpoint,
        suffix_len,
        format_bytes(suffix_bytes),
        powers_of_g.len(),
        total_affine_points,
        format_bytes(retained_bytes),
        snapshot
            .map(|memory| format_bytes(memory.resident_size as usize))
            .unwrap_or_else(|| "n/a".to_string()),
        snapshot
            .map(|memory| format_bytes(memory.phys_footprint as usize))
            .unwrap_or_else(|| "n/a".to_string()),
    );
}

impl<E: Pairing> StructuredReferenceString<E> for MultilinearUniversalParams<E> {
    type ProverParam = MultilinearProverParam<E>;
    type VerifierParam = MultilinearVerifierParam<E>;

    /// Extract the prover parameters from the public parameters.
    fn extract_prover_param(&self, supported_num_vars: usize) -> Self::ProverParam {
        let to_reduce = self.prover_param.num_vars - supported_num_vars;

        Self::ProverParam {
            powers_of_g: self.prover_param.powers_of_g[to_reduce..].to_vec(),
            g: self.prover_param.g,
            h: self.prover_param.h,
            num_vars: supported_num_vars,
        }
    }

    /// Extract the verifier parameters from the public parameters.
    fn extract_verifier_param(&self, supported_num_vars: usize) -> Self::VerifierParam {
        let to_reduce = self.prover_param.num_vars - supported_num_vars;
        Self::VerifierParam {
            num_vars: supported_num_vars,
            g: self.prover_param.g,
            h: self.prover_param.h,
            h_mask: self.h_mask[to_reduce..].to_vec(),
        }
    }

    /// Trim the universal parameters to specialize the public parameters
    /// for multilinear polynomials to the given `supported_num_vars`, and
    /// returns committer key and verifier key. `supported_num_vars` should
    /// be in range `1..=params.num_vars`
    fn trim(
        &self,
        supported_num_vars: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        if supported_num_vars > self.prover_param.num_vars {
            return Err(PCSError::InvalidParameters(format!(
                "SRS does not support target number of vars {}",
                supported_num_vars
            )));
        }

        let to_reduce = self.prover_param.num_vars - supported_num_vars;
        let ck = Self::ProverParam {
            powers_of_g: self.prover_param.powers_of_g[to_reduce..].to_vec(),
            g: self.prover_param.g,
            h: self.prover_param.h,
            num_vars: supported_num_vars,
        };
        let vk = Self::VerifierParam {
            num_vars: supported_num_vars,
            g: self.prover_param.g,
            h: self.prover_param.h,
            h_mask: self.h_mask[to_reduce..].to_vec(),
        };
        Ok((ck, vk))
    }

    /// Build SRS for testing.
    /// WARNING: THIS FUNCTION IS FOR TESTING PURPOSE ONLY.
    /// THE OUTPUT SRS SHOULD NOT BE USED IN PRODUCTION.
    fn gen_srs_for_testing<R: Rng>(rng: &mut R, num_vars: usize) -> Result<Self, PCSError> {
        if num_vars == 0 {
            return Err(PCSError::InvalidParameters(
                "constant polynomial not supported".to_string(),
            ));
        }

        let total_timer = start_timer!(|| "SRS generation");

        let pp_generation_timer = start_timer!(|| "Prover Param generation");

        let g = E::G1::rand(rng);
        let h = E::G2::rand(rng);

        let mut powers_of_g = Vec::new();

        let t: Vec<_> = (0..num_vars).map(|_| E::ScalarField::rand(rng)).collect();
        let max_scalars = 1 << num_vars;
        let g_batch_mul_preprocessing = BatchMulPreprocessing::<E::G1>::new(g, max_scalars);
        log_srs_setup_memory::<E>("srs_generation_start", None, &powers_of_g);

        let last = t[num_vars - 1];
        let mut current_eq = vec![E::ScalarField::one() - last, last];
        let mut suffix_start = num_vars - 1;

        loop {
            log_srs_setup_memory::<E>("suffix_scalar_ready", Some(current_eq.len()), &powers_of_g);
            log_srs_setup_memory::<E>("before_batch_mul", Some(current_eq.len()), &powers_of_g);
            let pp_k_g = Evaluations {
                evals: g_batch_mul_preprocessing.batch_mul(&current_eq),
            };
            log_srs_setup_memory::<E>("after_batch_mul", Some(current_eq.len()), &powers_of_g);
            // check correctness of pp_k_g
            let t_eval_0 = eq_eval(
                &vec![E::ScalarField::zero(); num_vars - suffix_start],
                &t[suffix_start..num_vars],
            )?;
            assert_eq!((g * t_eval_0).into(), pp_k_g.evals[0]);
            powers_of_g.push(pp_k_g);
            log_srs_setup_memory::<E>("after_powers_push", Some(current_eq.len()), &powers_of_g);

            if suffix_start == 0 {
                break;
            }

            suffix_start -= 1;
            current_eq = extend_eq_suffix_table(&current_eq, t[suffix_start]);
        }
        powers_of_g.reverse();
        let gg = Evaluations {
            evals: [g.into_affine()].to_vec(),
        };
        powers_of_g.push(gg);
        log_srs_setup_memory::<E>("srs_generation_end", None, &powers_of_g);

        let pp = Self::ProverParam {
            num_vars,
            g: g.into_affine(),
            h: h.into_affine(),
            powers_of_g,
        };

        end_timer!(pp_generation_timer);

        let vp_generation_timer = start_timer!(|| "VP generation");
        let h_batch_mul_preprocessing = BatchMulPreprocessing::<E::G2>::new(h, num_vars);
        let h_mask = h_batch_mul_preprocessing.batch_mul(&t);
        end_timer!(vp_generation_timer);
        end_timer!(total_timer);
        Ok(Self {
            prover_param: pp,
            h_mask,
        })
    }
}

#[cfg(test)]
/// fix first `pad` variables of `poly` represented in evaluation form to zero
fn remove_dummy_variable<F: Field>(poly: &[F], pad: usize) -> Result<Vec<F>, PCSError> {
    if pad == 0 {
        return Ok(poly.to_vec());
    }
    if !poly.len().is_power_of_two() {
        return Err(PCSError::InvalidParameters(
            "Size of polynomial should be power of two.".to_string(),
        ));
    }
    let nv = ark_std::log2(poly.len()) as usize - pad;
    Ok((0..(1 << nv)).map(|x| poly[x << pad]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::multilinear_kzg::util::eq_extension;
    use ark_bls12_381::Bls12_381;
    use ark_ff::PrimeField;
    use ark_poly::DenseMultilinearExtension;
    use ark_std::collections::LinkedList;
    use ark_std::test_rng;
    use core::iter::FromIterator;
    type E = Bls12_381;

    fn direct_eq_suffix_tables<F: Field>(t: &[F]) -> Vec<Vec<F>> {
        let mut suffix_tables = Vec::with_capacity(t.len());
        let mut current = vec![F::one() - t[t.len() - 1], t[t.len() - 1]];
        suffix_tables.push(current.clone());

        for &ti in t[..t.len() - 1].iter().rev() {
            current = extend_eq_suffix_table(&current, ti);
            suffix_tables.push(current.clone());
        }

        suffix_tables.reverse();
        suffix_tables
    }

    fn legacy_eq_suffix_tables<F: PrimeField>(t: &[F]) -> Result<Vec<Vec<F>>, PCSError> {
        let mut eq: LinkedList<DenseMultilinearExtension<F>> =
            LinkedList::from_iter(eq_extension(t));
        let mut eq_arr = LinkedList::new();
        let mut base = eq.pop_back().unwrap().evaluations;

        for i in (0..t.len()).rev() {
            eq_arr.push_front(remove_dummy_variable(&base, i)?);
            if i != 0 {
                let mul = eq.pop_back().unwrap().evaluations;
                base = base
                    .into_iter()
                    .zip(mul.into_iter())
                    .map(|(a, b)| a * b)
                    .collect();
            }
        }

        Ok(eq_arr.into_iter().collect())
    }

    #[test]
    fn test_srs_gen() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in 4..10 {
            let _ = MultilinearUniversalParams::<E>::gen_srs_for_testing(&mut rng, nv)?;
        }

        Ok(())
    }

    #[test]
    fn test_build_eq_suffix_tables_matches_legacy_construction() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in 1..8 {
            let t: Vec<_> = (0..nv)
                .map(|_| <E as Pairing>::ScalarField::rand(&mut rng))
                .collect();
            let direct = direct_eq_suffix_tables(&t);
            let legacy = legacy_eq_suffix_tables(&t)?;
            assert_eq!(direct, legacy, "suffix tables differ for nv={nv}");
        }

        Ok(())
    }
}
