// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Sumcheck based batch opening and verify commitment.
// TODO: refactoring this code to somewhere else
// currently IOP depends on PCS because perm check requires commitment.
// The sumcheck based batch opening therefore cannot stay in the PCS repo --
// which creates a cyclic dependency.

use crate::{
    pcs::{
        multilinear_kzg::util::eq_eval,
        prelude::{Commitment, PCSError},
        PolynomialCommitmentScheme,
    },
    poly_iop::{prelude::SumCheck, PolyIOP},
    IOPProof,
};
use arithmetic::{build_eq_x_r_vec, DenseMultilinearExtension, VPAuxInfo, VirtualPolynomial};
use ark_ec::{pairing::Pairing, scalar_mul::variable_base::VariableBaseMSM, CurveGroup};
use ark_ff::PrimeField;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_iter, cfg_iter_mut, end_timer, log2, start_timer, One, Zero};
#[cfg(feature = "parallel")]
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use std::cell::RefCell;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;
use std::{collections::BTreeMap, marker::PhantomData, ops::Deref, sync::Arc};
use transcript::IOPTranscript;
use std::sync::atomic::{AtomicBool, Ordering};

thread_local! {
    static BATCH_OPEN_CONTEXT_STACK: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

static BATCH_OPEN_TIMING_STATS: OnceLock<Mutex<BTreeMap<String, BatchOpenTimingAccumulator>>> =
    OnceLock::new();
static BATCH_OPEN_TIMING_ENABLED: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Debug)]
pub struct BatchOpenTimingStat {
    pub label: String,
    pub total_ms: f64,
    pub count: usize,
}

#[derive(Default)]
struct BatchOpenTimingAccumulator {
    total_ms: f64,
    count: usize,
}

#[derive(Debug)]
pub struct BatchOpenContextGuard {
    active: bool,
}

impl Drop for BatchOpenContextGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        BATCH_OPEN_CONTEXT_STACK.with(|stack| {
            stack.borrow_mut().pop();
        });
    }
}

pub fn scoped_batch_open_context(label: impl Into<String>) -> BatchOpenContextGuard {
    BATCH_OPEN_CONTEXT_STACK.with(|stack| {
        stack.borrow_mut().push(label.into());
    });
    BatchOpenContextGuard { active: true }
}

pub fn reset_batch_open_timing_stats() {
    let Ok(mut stats) = batch_open_timing_stats_store().lock() else {
        return;
    };
    stats.clear();
}

pub fn set_batch_open_timing_enabled(enabled: bool) {
    BATCH_OPEN_TIMING_ENABLED.store(enabled, Ordering::Relaxed);
}

fn batch_open_timing_enabled() -> bool {
    BATCH_OPEN_TIMING_ENABLED.load(Ordering::Relaxed)
}

pub fn batch_open_timing_stats() -> Vec<BatchOpenTimingStat> {
    let Ok(stats) = batch_open_timing_stats_store().lock() else {
        return Vec::new();
    };
    let mut snapshot = stats
        .iter()
        .map(|(label, stat)| BatchOpenTimingStat {
            label: label.clone(),
            total_ms: stat.total_ms,
            count: stat.count,
        })
        .collect::<Vec<_>>();
    snapshot.sort_by(|a, b| {
        b.total_ms
            .partial_cmp(&a.total_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.label.cmp(&b.label))
    });
    snapshot
}

fn batch_open_timing_stats_store() -> &'static Mutex<BTreeMap<String, BatchOpenTimingAccumulator>> {
    BATCH_OPEN_TIMING_STATS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn current_batch_open_context() -> Option<String> {
    BATCH_OPEN_CONTEXT_STACK.with(|stack| stack.borrow().last().cloned())
}

fn record_batch_open_timing(label: &str, elapsed_ms: f64) {
    if !batch_open_timing_enabled() {
        return;
    }
    let Ok(mut stats) = batch_open_timing_stats_store().lock() else {
        return;
    };
    let entry = stats.entry(label.to_string()).or_default();
    entry.total_ms += elapsed_ms;
    entry.count += 1;
}

fn batch_open_record_stage(stage: &str, elapsed_ms: f64) {
    let label = current_batch_open_context()
        .map(|context| format!("{context} / {stage}"))
        .unwrap_or_else(|| format!("pcs batch_open / {stage}"));
    record_batch_open_timing(&label, elapsed_ms);
}

fn batch_open_log_enabled() -> bool {
    std::env::var_os("AJTAI_PCS_BATCH_LOG").is_some()
        || std::env::var_os("AJTAI_SETUP_MEM_LOG").is_some()
}

fn batch_open_format_bytes(bytes: usize) -> String {
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

fn batch_open_current_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
                return Some(kb.saturating_mul(1024));
            }
        }
    }

    let pid = std::process::id().to_string();
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let rss_kb = String::from_utf8(output.stdout)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()?;
    Some(rss_kb.saturating_mul(1024))
}

fn batch_open_note(message: impl AsRef<str>) {
    if batch_open_log_enabled() {
        eprintln!("[mlkzg-batch-open] {}", message.as_ref());
    }
}

fn batch_open_checkpoint<E: Pairing>(
    label: &str,
    num_var: usize,
    point_count: usize,
    unique_point_count: usize,
    started_at: &Instant,
) {
    if !batch_open_log_enabled() {
        return;
    }
    let eval_count = 1usize.checked_shl(num_var as u32).unwrap_or(usize::MAX);
    let point_bytes = point_count
        .saturating_mul(num_var)
        .saturating_mul(core::mem::size_of::<E::ScalarField>());
    let per_tilde_eq_bytes = eval_count.saturating_mul(core::mem::size_of::<E::ScalarField>());
    let total_tilde_eq_bytes = unique_point_count.saturating_mul(per_tilde_eq_bytes);
    let rss = batch_open_current_rss_bytes()
        .map(|value| batch_open_format_bytes(value as usize))
        .unwrap_or_else(|| "unavailable".to_string());
    batch_open_note(format!(
        "checkpoint={} elapsed_ms={:.3} num_var={} point_count={} unique_points={} point_bytes={} per_tilde_eq_bytes={} total_tilde_eq_bytes={} rss={}",
        label,
        started_at.elapsed().as_secs_f64() * 1_000.0,
        num_var,
        point_count,
        unique_point_count,
        batch_open_format_bytes(point_bytes),
        batch_open_format_bytes(per_tilde_eq_bytes),
        batch_open_format_bytes(total_tilde_eq_bytes),
        rss,
    ));
}

fn decode_boolean_prefix_index<F: PrimeField>(point: &[F], prefix_len: usize) -> usize {
    let mut index = 0usize;
    for (bit, value) in point.iter().take(prefix_len).enumerate() {
        debug_assert!(value.is_zero() || value.is_one());
        if value.is_one() {
            index |= 1usize << bit;
        }
    }
    index
}

fn leading_boolean_prefix_len<F: PrimeField>(point: &[F]) -> usize {
    point
        .iter()
        .take_while(|value| value.is_zero() || value.is_one())
        .count()
}

fn build_weighted_eq_mle<F: PrimeField>(
    num_var: usize,
    weighted_points: &[(F, &[F])],
) -> Result<Arc<DenseMultilinearExtension<F>>, PCSError> {
    let dense_scan_timer = start_timer!(|| {
        format!(
            "multi_open_dense_scan build_weighted_eq num_var={} terms={}",
            num_var,
            weighted_points.len()
        )
    });
    let domain_size = 1usize.checked_shl(num_var as u32).ok_or_else(|| {
        PCSError::InvalidParameters("too many variables for equality table".to_string())
    })?;
    let mut evaluations = vec![F::zero(); domain_size];

    for (coeff, point) in weighted_points.iter().copied() {
        if coeff.is_zero() {
            continue;
        }
        let prefix_len = leading_boolean_prefix_len(point);
        if prefix_len == num_var {
            let index = decode_boolean_prefix_index(point, prefix_len);
            evaluations[index] += coeff;
            continue;
        }

        if prefix_len == 0 {
            let dense_eq = build_eq_x_r_vec(point)
                .map_err(|err| PCSError::InvalidParameters(err.to_string()))?;
            if coeff.is_one() {
                for (dst, src) in evaluations.iter_mut().zip(dense_eq.into_iter()) {
                    *dst += src;
                }
            } else {
                for (dst, src) in evaluations.iter_mut().zip(dense_eq.into_iter()) {
                    *dst += coeff * src;
                }
            }
            continue;
        }

        let suffix_eq = build_eq_x_r_vec(&point[prefix_len..])
            .map_err(|err| PCSError::InvalidParameters(err.to_string()))?;
        let prefix_index = decode_boolean_prefix_index(point, prefix_len);
        if coeff.is_one() {
            for (suffix_index, value) in suffix_eq.into_iter().enumerate() {
                evaluations[prefix_index + (suffix_index << prefix_len)] += value;
            }
        } else {
            for (suffix_index, value) in suffix_eq.into_iter().enumerate() {
                evaluations[prefix_index + (suffix_index << prefix_len)] += coeff * value;
            }
        }
    }

    let result = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
        num_var,
        evaluations,
    ));
    end_timer!(dense_scan_timer);
    Ok(result)
}

fn fused_scaled_sum_dense_mles<'a, F, I>(
    num_var: usize,
    terms: I,
) -> Arc<DenseMultilinearExtension<F>>
where
    F: PrimeField,
    I: IntoIterator<Item = (F, &'a DenseMultilinearExtension<F>)>,
{
    let dense_scan_timer =
        start_timer!(|| format!("multi_open_dense_scan build_g_prime num_var={}", num_var));
    let mut acc: Option<Vec<F>> = None;
    let mut eval_len = None;

    for (coeff, poly) in terms {
        debug_assert_eq!(
            poly.num_vars, num_var,
            "all merged multilinear polynomials must have the same number of variables",
        );
        let poly_len = poly.evaluations.len();
        if eval_len.is_none() {
            eval_len = Some(poly_len);
        }
        if coeff.is_zero() {
            continue;
        }

        match acc.as_mut() {
            None => {
                let initial = if coeff.is_one() {
                    poly.evaluations.clone()
                } else {
                    cfg_iter!(poly.evaluations)
                        .map(|value| *value * coeff)
                        .collect()
                };
                acc = Some(initial);
            },
            Some(acc) => {
                debug_assert_eq!(
                    acc.len(),
                    poly_len,
                    "all merged multilinear polynomials must have matching evaluation lengths",
                );
                if coeff.is_one() {
                    cfg_iter_mut!(acc)
                        .zip(cfg_iter!(poly.evaluations))
                        .for_each(|(dst, src)| {
                            *dst += src;
                        });
                } else {
                    cfg_iter_mut!(acc)
                        .zip(cfg_iter!(poly.evaluations))
                        .for_each(|(dst, src)| {
                            *dst += *src * coeff;
                        });
                }
            },
        }
    }

    let evaluations = acc.unwrap_or_else(|| vec![F::zero(); eval_len.unwrap_or(1usize << num_var)]);
    let result = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
        num_var,
        evaluations,
    ));
    end_timer!(dense_scan_timer);
    result
}

#[derive(Clone, Debug, Default, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchProof<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    /// A sum check proof proving tilde g's sum
    pub(crate) sum_check_proof: IOPProof<E::ScalarField>,
    /// f_i(point_i)
    pub f_i_eval_at_point_i: Vec<E::ScalarField>,
    /// proof for g'(a_2)
    pub(crate) g_prime_proof: PCS::Proof,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchProofWithoutEvals<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    /// A sum check proof proving tilde g's sum
    pub(crate) sum_check_proof: IOPProof<E::ScalarField>,
    /// proof for g'(a_2)
    pub(crate) g_prime_proof: PCS::Proof,
}

impl<E, PCS> BatchProof<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub fn without_evaluations(&self) -> BatchProofWithoutEvals<E, PCS> {
        BatchProofWithoutEvals {
            sum_check_proof: self.sum_check_proof.clone(),
            g_prime_proof: self.g_prime_proof.clone(),
        }
    }
}

impl<E, PCS> BatchProofWithoutEvals<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub fn with_evaluations(&self, evals: Vec<E::ScalarField>) -> BatchProof<E, PCS> {
        BatchProof {
            sum_check_proof: self.sum_check_proof.clone(),
            f_i_eval_at_point_i: evals,
            g_prime_proof: self.g_prime_proof.clone(),
        }
    }
}

/// Steps:
/// 1. get challenge point t from transcript
/// 2. build eq(t,i) for i in [0..k]
/// 3. build \tilde g_i(b) = eq(t, i) * f_i(b)
/// 4. compute \tilde eq_i(b) = eq(b, point_i)
/// 5. run sumcheck on \sum_i=1..k \tilde eq_i * \tilde g_i
/// 6. build g'(X) = \sum_i=1..k \tilde eq_i(a2) * \tilde g_i(X) where (a2) is
///    the sumcheck's point 7. open g'(X) at point (a2)
pub(crate) fn multi_open_internal<E, PCS>(
    prover_param: &PCS::ProverParam,
    polynomials: &[PCS::Polynomial],
    points: &[PCS::Point],
    evals: &[PCS::Evaluation],
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<BatchProof<E, PCS>, PCSError>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Point = Vec<E::ScalarField>,
        Evaluation = E::ScalarField,
    >,
{
    let open_timer = start_timer!(|| format!("multi_open_total num_queries={}", points.len()));
    let batch_started_at = Instant::now();
    let record_stage = |stage: &str, started_at: Instant| {
        batch_open_record_stage(stage, started_at.elapsed().as_secs_f64() * 1_000.0);
    };
    let preprocess_started_at = Instant::now();
    let preprocess_timer =
        start_timer!(|| format!("multi_open_preprocess num_queries={}", points.len()));
    batch_open_note(format!(
        "start point_count={} polynomial_count={} eval_count={} rss={}",
        points.len(),
        polynomials.len(),
        evals.len(),
        batch_open_current_rss_bytes()
            .map(|value| batch_open_format_bytes(value as usize))
            .unwrap_or_else(|| "unavailable".to_string()),
    ));
    for eval_point in points.iter() {
        transcript.append_serializable_element(b"eval_point", eval_point)?;
    }
    for eval in evals.iter() {
        transcript.append_field_element(b"eval", eval)?;
    }

    // TODO: sanity checks
    let num_var = polynomials[0].num_vars;
    let k = polynomials.len();
    let ell = log2(k) as usize;

    // challenge point t
    let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;

    // eq(t, i) for i in [0..k]
    let eq_t_i_list = build_eq_x_r_vec(t.as_ref())?;
    end_timer!(preprocess_timer);
    record_stage("preprocess", preprocess_started_at);

    // \tilde g_i(b) = eq(t, i) * f_i(b)
    let timer = start_timer!(|| format!("multi_open_group_queries num_queries={}", points.len()));
    let stage_started_at = Instant::now();
    // combine the polynomials that have same opening point first to reduce the
    // cost of sum check later.
    let point_indices = points
        .iter()
        .fold(BTreeMap::<_, _>::new(), |mut indices, point| {
            let idx = indices.len();
            indices.entry(point).or_insert(idx);
            indices
        });
    let deduped_points =
        BTreeMap::from_iter(point_indices.iter().map(|(point, idx)| (*idx, *point)))
            .into_values()
            .collect::<Vec<_>>();
    let point_bucket_indices = points
        .iter()
        .map(|point| point_indices[point])
        .collect::<Vec<_>>();
    let poly_indices =
        polynomials
            .iter()
            .fold(BTreeMap::<usize, usize>::new(), |mut indices, poly| {
                let key = Arc::as_ptr(poly) as usize;
                let idx = indices.len();
                indices.entry(key).or_insert(idx);
                indices
            });
    let unique_polys = {
        let mut slots = vec![None; poly_indices.len()];
        for poly in polynomials.iter() {
            let key = Arc::as_ptr(poly) as usize;
            let idx = poly_indices[&key];
            if slots[idx].is_none() {
                slots[idx] = Some(poly.clone());
            }
        }
        slots
            .into_iter()
            .map(|slot| slot.expect("poly index slot must be populated"))
            .collect::<Vec<_>>()
    };
    let mut bucket_sizes = vec![0usize; deduped_points.len()];
    for &point_idx in point_bucket_indices.iter() {
        bucket_sizes[point_idx] += 1;
    }
    let max_bucket = bucket_sizes.iter().copied().max().unwrap_or(0);
    let singleton_buckets = bucket_sizes.iter().filter(|&&size| size == 1).count();
    let mut poly_point_coeffs = vec![BTreeMap::<usize, E::ScalarField>::new(); unique_polys.len()];
    for (((poly, coeff), point_idx), point) in polynomials
        .iter()
        .zip(eq_t_i_list.iter())
        .zip(point_bucket_indices.iter())
        .zip(points.iter())
    {
        let poly_key = Arc::as_ptr(poly) as usize;
        let poly_idx = poly_indices[&poly_key];
        debug_assert_eq!(*point_idx, point_indices[point]);
        *poly_point_coeffs[poly_idx]
            .entry(*point_idx)
            .or_insert_with(E::ScalarField::zero) += *coeff;
    }
    batch_open_note(format!(
        "merge_plan unique_points={} unique_polynomials={} max_bucket={} singleton_buckets={} dense_merge_materialization_skipped=true",
        deduped_points.len(),
        unique_polys.len(),
        max_bucket,
        singleton_buckets,
    ));
    batch_open_note(format!(
        "multi_open_metadata context={} num_queries={} num_unique_points={} num_polynomials={} num_unique_polynomials={} num_vars={}",
        current_batch_open_context().unwrap_or_else(|| "unknown".to_string()),
        points.len(),
        deduped_points.len(),
        polynomials.len(),
        unique_polys.len(),
        num_var,
    ));
    batch_open_checkpoint::<E>(
        "deduped_points_ready",
        num_var,
        points.len(),
        deduped_points.len(),
        &batch_started_at,
    );
    end_timer!(timer);
    record_stage("group_queries", stage_started_at);
    batch_open_checkpoint::<E>(
        "merged_tilde_g_ready",
        num_var,
        points.len(),
        deduped_points.len(),
        &batch_started_at,
    );

    let timer =
        start_timer!(|| format!("multi_open_build_eq unique_points={}", deduped_points.len()));
    let stage_started_at = Instant::now();
    batch_open_checkpoint::<E>(
        "tilde_eq_build_start",
        num_var,
        points.len(),
        deduped_points.len(),
        &batch_started_at,
    );
    let single_point_eq = if deduped_points.len() == 1 {
        batch_open_note(
            "single_point_fast_path=true reusing one eq table across all batch-open terms",
        );
        Some(build_weighted_eq_mle(
            num_var,
            &[(E::ScalarField::one(), deduped_points[0].as_slice())],
        )?)
    } else {
        None
    };
    let weighted_tilde_eqs = if single_point_eq.is_some() {
        Vec::new()
    } else {
        poly_point_coeffs
            .iter()
            .map(|point_coeffs| {
                let weighted_points = point_coeffs
                    .iter()
                    .filter_map(|(point_idx, coeff)| {
                        (!coeff.is_zero()).then_some((*coeff, deduped_points[*point_idx].as_slice()))
                    })
                    .collect::<Vec<_>>();
                build_weighted_eq_mle(num_var, &weighted_points)
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    end_timer!(timer);
    record_stage("build_eq", stage_started_at);
    batch_open_checkpoint::<E>(
        "tilde_eq_build_done",
        num_var,
        points.len(),
        deduped_points.len(),
        &batch_started_at,
    );

    // built the virtual polynomial for SumCheck
    let timer = start_timer!(|| format!("multi_open_build_virtual_poly num_vars={}", num_var));
    let sumcheck_total_started_at = Instant::now();
    batch_open_checkpoint::<E>(
        "sumcheck_build_start",
        num_var,
        points.len(),
        deduped_points.len(),
        &batch_started_at,
    );

    let step = start_timer!(|| "multi_open_add_mle");
    let add_mles_started_at = Instant::now();
    let mut sum_check_vp = VirtualPolynomial::new(num_var);
    if let Some(shared_eq) = single_point_eq {
        for (poly, point_coeffs) in unique_polys.iter().zip(poly_point_coeffs.iter()) {
            debug_assert!(point_coeffs.len() <= 1);
            let coeff = point_coeffs
                .get(&0)
                .copied()
                .unwrap_or_else(E::ScalarField::zero);
            if coeff.is_zero() {
                continue;
            }
            sum_check_vp.add_mle_list([poly.clone(), shared_eq.clone()], coeff)?;
        }
    } else {
        for (poly, weighted_tilde_eq) in unique_polys.iter().zip(weighted_tilde_eqs.iter()) {
            sum_check_vp.add_mle_list(
                [poly.clone(), weighted_tilde_eq.clone()],
                E::ScalarField::one(),
            )?;
        }
    }
    end_timer!(step);
    record_stage("sumcheck_build", add_mles_started_at);
    batch_open_checkpoint::<E>(
        "sumcheck_add_mles_done",
        num_var,
        points.len(),
        deduped_points.len(),
        &batch_started_at,
    );

    let sumcheck_prove_started_at = Instant::now();
    let sumcheck_timer = start_timer!(|| format!("multi_open_sumcheck_prove num_vars={}", num_var));
    let proof = match <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::prove(
        &sum_check_vp,
        transcript,
    ) {
        Ok(p) => p,
        Err(_e) => {
            // cannot wrap IOPError with PCSError due to cyclic dependency
            return Err(PCSError::InvalidProver(
                "Sumcheck in batch proving Failed".to_string(),
            ));
        },
    };

    end_timer!(sumcheck_timer);
    end_timer!(timer);
    record_stage("sumcheck_prove", sumcheck_prove_started_at);
    record_stage("build_virtual_poly", sumcheck_total_started_at);
    batch_open_checkpoint::<E>(
        "sumcheck_prove_done",
        num_var,
        points.len(),
        deduped_points.len(),
        &batch_started_at,
    );

    // a2 := sumcheck's point
    let a2 = &proof.point[..num_var];

    // build g'(X) = \sum_i=1..k eq(a2, point_i) * eq(t, <i>) * f_i(X)
    let step = start_timer!(|| format!("multi_open_build_g_prime num_queries={}", points.len()));
    let g_prime_started_at = Instant::now();
    let eq_a2_per_point = deduped_points
        .iter()
        .map(|point| eq_eval(a2, point))
        .collect::<Result<Vec<_>, _>>()?;
    let g_prime_terms = unique_polys
        .iter()
        .zip(poly_point_coeffs.iter())
        .filter_map(|(poly, point_coeffs)| {
            let coeff = point_coeffs.iter().fold(E::ScalarField::zero(), |acc, (point_idx, w)| {
                acc + (eq_a2_per_point[*point_idx] * *w)
            });
            (!coeff.is_zero()).then_some((coeff, poly.deref()))
        })
        .collect::<Vec<_>>();
    batch_open_note(format!(
        "g_prime_metadata context={} num_terms={} g_prime_num_vars={} num_queries={} num_unique_points={} num_unique_polynomials={}",
        current_batch_open_context().unwrap_or_else(|| "unknown".to_string()),
        g_prime_terms.len(),
        num_var,
        points.len(),
        deduped_points.len(),
        unique_polys.len(),
    ));
    let g_prime = fused_scaled_sum_dense_mles(num_var, g_prime_terms);
    end_timer!(step);
    record_stage("g_prime_construction", g_prime_started_at);
    batch_open_checkpoint::<E>(
        "g_prime_ready",
        num_var,
        points.len(),
        deduped_points.len(),
        &batch_started_at,
    );

    let step = start_timer!(|| format!("multi_open_open_g_prime num_vars={}", num_var));
    let pcs_open_started_at = Instant::now();
    let (g_prime_proof, _g_prime_eval) = PCS::open(prover_param, &g_prime, a2.to_vec().as_ref())?;
    // assert_eq!(g_prime_eval, tilde_g_eval);
    end_timer!(step);
    record_stage("pcs_open", pcs_open_started_at);
    batch_open_checkpoint::<E>(
        "pcs_open_done",
        num_var,
        points.len(),
        deduped_points.len(),
        &batch_started_at,
    );

    let step = start_timer!(|| "evaluate fi(pi)");
    end_timer!(step);
    end_timer!(open_timer);
    record_stage("total", batch_started_at);
    batch_open_checkpoint::<E>(
        "done",
        num_var,
        points.len(),
        deduped_points.len(),
        &batch_started_at,
    );

    Ok(BatchProof {
        sum_check_proof: proof,
        f_i_eval_at_point_i: evals.to_vec(),
        g_prime_proof,
    })
}

/// Steps:
/// 1. get challenge point t from transcript
/// 2. build g' commitment
/// 3. ensure \sum_i eq(a2, point_i) * eq(t, <i>) * f_i_evals matches the sum
///    via SumCheck verification 4. verify commitment
pub(crate) fn batch_verify_internal<E, PCS>(
    verifier_param: &PCS::VerifierParam,
    f_i_commitments: &[Commitment<E>],
    points: &[PCS::Point],
    proof: &BatchProof<E, PCS>,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<bool, PCSError>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Point = Vec<E::ScalarField>,
        Evaluation = E::ScalarField,
        Commitment = Commitment<E>,
    >,
{
    let open_timer = start_timer!(|| "batch verification");
    for eval_point in points.iter() {
        transcript.append_serializable_element(b"eval_point", eval_point)?;
    }
    for eval in proof.f_i_eval_at_point_i.iter() {
        transcript.append_field_element(b"eval", eval)?;
    }

    // TODO: sanity checks

    let k = f_i_commitments.len();
    let ell = log2(k) as usize;
    let num_var = proof.sum_check_proof.point.len();

    // challenge point t
    let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;

    // sum check point (a2)
    let a2 = &proof.sum_check_proof.point[..num_var];

    // build g' commitment
    let step = start_timer!(|| "build homomorphic commitment");
    let eq_t_list = build_eq_x_r_vec(t.as_ref())?;

    let mut scalars = vec![];
    let mut bases = vec![];

    for (i, point) in points.iter().enumerate() {
        let eq_i_a2 = eq_eval(a2, point)?;
        scalars.push(eq_i_a2 * eq_t_list[i]);
        bases.push(f_i_commitments[i].0);
    }
    let g_prime_commit = E::G1::msm_unchecked(&bases, &scalars);
    end_timer!(step);

    // ensure \sum_i eq(t, <i>) * f_i_evals matches the sum via SumCheck
    let mut sum = E::ScalarField::zero();
    for (i, &e) in eq_t_list.iter().enumerate().take(k) {
        sum += e * proof.f_i_eval_at_point_i[i];
    }
    let aux_info = VPAuxInfo {
        max_degree: 2,
        num_variables: num_var,
        phantom: PhantomData,
    };
    let subclaim = match <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::verify(
        sum,
        &proof.sum_check_proof,
        &aux_info,
        transcript,
    ) {
        Ok(p) => p,
        Err(_e) => {
            // cannot wrap IOPError with PCSError due to cyclic dependency
            return Err(PCSError::InvalidProver(
                "Sumcheck in batch verification failed".to_string(),
            ));
        },
    };
    let tilde_g_eval = subclaim.expected_evaluation;

    // verify commitment
    let res = PCS::verify(
        verifier_param,
        &Commitment(g_prime_commit.into_affine()),
        a2.to_vec().as_ref(),
        &tilde_g_eval,
        &proof.g_prime_proof,
    )?;

    end_timer!(open_timer);
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::{
        prelude::{MultilinearKzgPCS, MultilinearUniversalParams},
        StructuredReferenceString,
    };
    use arithmetic::get_batched_nv;
    use ark_bls12_381::Bls12_381 as E;
    use ark_ec::pairing::Pairing;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension, Polynomial};
    use ark_std::{rand::Rng, test_rng, vec::Vec, UniformRand};

    type Fr = <E as Pairing>::ScalarField;

    #[test]
    fn fused_scaled_sum_dense_mles_matches_naive_merge() {
        let mut rng = test_rng();
        let nv = 5usize;
        let polys = (0..3)
            .map(|_| DenseMultilinearExtension::<Fr>::rand(nv, &mut rng))
            .collect::<Vec<_>>();
        let coeffs = (0..polys.len())
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();

        let mut naive = Arc::new(DenseMultilinearExtension::zero());
        for (coeff, poly) in coeffs.iter().zip(polys.iter()) {
            *Arc::make_mut(&mut naive) += (*coeff, poly);
        }

        let fused = fused_scaled_sum_dense_mles(nv, coeffs.iter().copied().zip(polys.iter()));
        assert_eq!(fused.num_vars, nv);
        assert_eq!(fused.evaluations, naive.evaluations);
    }

    #[test]
    fn weighted_eq_builder_matches_dense_sum() -> Result<(), PCSError> {
        let num_vars = 5usize;
        let points = vec![
            vec![
                Fr::one(),
                Fr::zero(),
                Fr::from(3u64),
                Fr::from(5u64),
                Fr::from(7u64),
            ],
            vec![
                Fr::zero(),
                Fr::one(),
                Fr::from(11u64),
                Fr::from(13u64),
                Fr::from(17u64),
            ],
        ];
        let coeffs = vec![Fr::from(19u64), Fr::from(23u64)];
        let weighted_points = coeffs
            .iter()
            .copied()
            .zip(points.iter().map(|point| point.as_slice()))
            .collect::<Vec<_>>();

        let built = build_weighted_eq_mle(num_vars, &weighted_points)?;

        let mut expected = vec![Fr::zero(); 1usize << num_vars];
        for (coeff, point) in coeffs.iter().zip(points.iter()) {
            let eq =
                build_eq_x_r_vec(point).map_err(|e| PCSError::InvalidParameters(e.to_string()))?;
            for (dst, src) in expected.iter_mut().zip(eq.into_iter()) {
                *dst += *coeff * src;
            }
        }

        assert_eq!(built.num_vars, num_vars);
        assert_eq!(built.evaluations, expected);
        Ok(())
    }

    fn test_multi_open_helper<R: Rng>(
        ml_params: &MultilinearUniversalParams<E>,
        polys: &[Arc<DenseMultilinearExtension<Fr>>],
        rng: &mut R,
    ) -> Result<(), PCSError> {
        let merged_nv = get_batched_nv(polys[0].num_vars(), polys.len());
        let (ml_ck, ml_vk) = ml_params.trim(merged_nv)?;

        let mut points = Vec::new();
        for poly in polys.iter() {
            let point = (0..poly.num_vars())
                .map(|_| Fr::rand(rng))
                .collect::<Vec<Fr>>();
            points.push(point);
        }

        let evals = polys
            .iter()
            .zip(points.iter())
            .map(|(f, p)| f.evaluate(p))
            .collect::<Vec<_>>();

        let commitments = polys
            .iter()
            .map(|poly| MultilinearKzgPCS::commit(ml_ck.clone(), poly).unwrap())
            .collect::<Vec<_>>();

        let mut transcript = IOPTranscript::new("test transcript".as_ref());
        transcript.append_field_element("init".as_ref(), &Fr::zero())?;

        let batch_proof = multi_open_internal::<E, MultilinearKzgPCS<E>>(
            &ml_ck,
            polys,
            &points,
            &evals,
            &mut transcript,
        )?;

        // good path
        let mut transcript = IOPTranscript::new("test transcript".as_ref());
        transcript.append_field_element("init".as_ref(), &Fr::zero())?;
        assert!(batch_verify_internal::<E, MultilinearKzgPCS<E>>(
            &ml_vk,
            &commitments,
            &points,
            &batch_proof,
            &mut transcript
        )?);

        let stripped_batch_proof = batch_proof.without_evaluations();
        let restored_batch_proof = stripped_batch_proof.with_evaluations(evals.clone());
        let mut transcript = IOPTranscript::new("test transcript".as_ref());
        transcript.append_field_element("init".as_ref(), &Fr::zero())?;
        assert!(batch_verify_internal::<E, MultilinearKzgPCS<E>>(
            &ml_vk,
            &commitments,
            &points,
            &restored_batch_proof,
            &mut transcript
        )?);

        Ok(())
    }

    fn test_multi_open_same_point_helper<R: Rng>(
        ml_params: &MultilinearUniversalParams<E>,
        polys: &[Arc<DenseMultilinearExtension<Fr>>],
        rng: &mut R,
    ) -> Result<(), PCSError> {
        let merged_nv = get_batched_nv(polys[0].num_vars(), polys.len());
        let (ml_ck, ml_vk) = ml_params.trim(merged_nv)?;

        let shared_point = (0..polys[0].num_vars())
            .map(|_| Fr::rand(rng))
            .collect::<Vec<Fr>>();
        let points = vec![shared_point.clone(); polys.len()];

        let evals = polys
            .iter()
            .map(|poly| poly.evaluate(&shared_point))
            .collect::<Vec<_>>();

        let commitments = polys
            .iter()
            .map(|poly| MultilinearKzgPCS::commit(ml_ck.clone(), poly).unwrap())
            .collect::<Vec<_>>();

        let mut transcript = IOPTranscript::new("test transcript".as_ref());
        transcript.append_field_element("init".as_ref(), &Fr::zero())?;

        let batch_proof = multi_open_internal::<E, MultilinearKzgPCS<E>>(
            &ml_ck,
            polys,
            &points,
            &evals,
            &mut transcript,
        )?;

        let mut transcript = IOPTranscript::new("test transcript".as_ref());
        transcript.append_field_element("init".as_ref(), &Fr::zero())?;
        assert!(batch_verify_internal::<E, MultilinearKzgPCS<E>>(
            &ml_vk,
            &commitments,
            &points,
            &batch_proof,
            &mut transcript
        )?);

        Ok(())
    }

    #[test]
    fn test_multi_open_internal() -> Result<(), PCSError> {
        let mut rng = test_rng();

        let ml_params = MultilinearUniversalParams::<E>::gen_srs_for_testing(&mut rng, 20)?;
        for num_poly in 5..6 {
            for nv in 15..16 {
                let polys1: Vec<_> = (0..num_poly)
                    .map(|_| Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)))
                    .collect();
                test_multi_open_helper(&ml_params, &polys1, &mut rng)?;
            }
        }

        Ok(())
    }

    #[test]
    fn test_multi_open_internal_same_point() -> Result<(), PCSError> {
        let mut rng = test_rng();

        let ml_params = MultilinearUniversalParams::<E>::gen_srs_for_testing(&mut rng, 20)?;
        for num_poly in 5..6 {
            for nv in 15..16 {
                let polys: Vec<_> = (0..num_poly)
                    .map(|_| Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)))
                    .collect();
                test_multi_open_same_point_helper(&ml_params, &polys, &mut rng)?;
            }
        }

        Ok(())
    }
}
