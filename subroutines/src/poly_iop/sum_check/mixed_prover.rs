use super::SumCheckProver;
use crate::poly_iop::{
    errors::PolyIOPErrors,
    structs::{IOPProof, IOPProverState},
};
use arithmetic::{VPAuxInfo, VirtualPolynomial};
use ark_ff::PrimeField;
use ark_std::{end_timer, start_timer, vec::Vec};
use std::{cmp::max, marker::PhantomData};
use transcript::IOPTranscript;

#[derive(Clone, Debug, PartialEq, Eq)]
struct SparseEntry<F: PrimeField> {
    index: usize,
    value: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SparsePair<F: PrimeField> {
    parent_index: usize,
    left: F,
    right: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseSupportProduct<F: PrimeField> {
    coefficient: F,
    num_vars: usize,
    left_entries: Vec<SparseEntry<F>>,
    right_entries: Vec<SparseEntry<F>>,
}

impl<F: PrimeField> SparseSupportProduct<F> {
    pub fn new(
        num_vars: usize,
        coefficient: F,
        left_entries: impl IntoIterator<Item = (usize, F)>,
        right_entries: impl IntoIterator<Item = (usize, F)>,
    ) -> Result<Self, PolyIOPErrors> {
        let domain_size = 1usize
            .checked_shl(num_vars as u32)
            .ok_or_else(|| PolyIOPErrors::InvalidParameters("too many variables".to_string()))?;
        Ok(Self {
            coefficient,
            num_vars,
            left_entries: normalize_sparse_entries(left_entries, domain_size)?,
            right_entries: normalize_sparse_entries(right_entries, domain_size)?,
        })
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn support_len_left(&self) -> usize {
        self.left_entries.len()
    }

    pub fn support_len_right(&self) -> usize {
        self.right_entries.len()
    }

    pub fn evaluate_factors_at_point(&self, point: &[F]) -> Result<(F, F), PolyIOPErrors> {
        if point.len() != self.num_vars {
            return Err(PolyIOPErrors::InvalidParameters(format!(
                "wrong number of variables {} vs {}",
                point.len(),
                self.num_vars
            )));
        }

        let mut state = self.clone();
        for challenge in point.iter().copied() {
            state.apply_challenge(challenge)?;
        }
        Ok((
            state.extract_constant(&state.left_entries)?,
            state.extract_constant(&state.right_entries)?,
        ))
    }

    fn round_evaluations(&self, max_degree: usize) -> Vec<F> {
        let left_pairs = build_sparse_pairs(&self.left_entries);
        let right_pairs = build_sparse_pairs(&self.right_entries);
        let eval_points = (0..=max_degree)
            .map(|i| F::from(i as u64))
            .collect::<Vec<_>>();
        let mut evaluations = vec![F::zero(); max_degree + 1];
        let mut left_cursor = 0usize;
        let mut right_cursor = 0usize;

        while left_cursor < left_pairs.len() || right_cursor < right_pairs.len() {
            let next_parent = match (left_pairs.get(left_cursor), right_pairs.get(right_cursor)) {
                (Some(left), Some(right)) => left.parent_index.min(right.parent_index),
                (Some(left), None) => left.parent_index,
                (None, Some(right)) => right.parent_index,
                (None, None) => unreachable!("loop guard ensures one pair remains"),
            };

            let left_pair = if left_pairs
                .get(left_cursor)
                .map(|pair| pair.parent_index == next_parent)
                .unwrap_or(false)
            {
                let pair = &left_pairs[left_cursor];
                left_cursor += 1;
                Some(pair)
            } else {
                None
            };
            let right_pair = if right_pairs
                .get(right_cursor)
                .map(|pair| pair.parent_index == next_parent)
                .unwrap_or(false)
            {
                let pair = &right_pairs[right_cursor];
                right_cursor += 1;
                Some(pair)
            } else {
                None
            };

            let left_zero = F::zero();
            let right_zero = F::zero();
            let (left_0, left_1) = left_pair
                .map(|pair| (pair.left, pair.right))
                .unwrap_or((left_zero, right_zero));
            let (right_0, right_1) = right_pair
                .map(|pair| (pair.left, pair.right))
                .unwrap_or((left_zero, right_zero));
            let left_step = left_1 - left_0;
            let right_step = right_1 - right_0;

            for (evaluation, point) in evaluations.iter_mut().zip(eval_points.iter()) {
                let left_value = left_0 + left_step * *point;
                let right_value = right_0 + right_step * *point;
                *evaluation += self.coefficient * left_value * right_value;
            }
        }

        evaluations
    }

    fn apply_challenge(&mut self, challenge: F) -> Result<(), PolyIOPErrors> {
        if self.num_vars == 0 {
            return Err(PolyIOPErrors::InvalidParameters(
                "cannot fix variables on a constant sparse support product".to_string(),
            ));
        }
        let left_pairs = build_sparse_pairs(&self.left_entries);
        let right_pairs = build_sparse_pairs(&self.right_entries);
        self.left_entries = collapse_pairs(left_pairs, challenge);
        self.right_entries = collapse_pairs(right_pairs, challenge);
        self.num_vars -= 1;
        Ok(())
    }

    fn extract_constant(&self, entries: &[SparseEntry<F>]) -> Result<F, PolyIOPErrors> {
        if self.num_vars != 0 {
            return Err(PolyIOPErrors::InvalidParameters(
                "sparse support product must be fully fixed before extracting a constant"
                    .to_string(),
            ));
        }

        match entries {
            [] => Ok(F::zero()),
            [entry] if entry.index == 0 => Ok(entry.value),
            [entry] => Err(PolyIOPErrors::InvalidParameters(format!(
                "unexpected constant-table index {}",
                entry.index
            ))),
            _ => Err(PolyIOPErrors::InvalidParameters(
                "constant sparse table has multiple support entries".to_string(),
            )),
        }
    }
}

pub fn prove_mixed_sparse_products<F: PrimeField>(
    dense_poly: &VirtualPolynomial<F>,
    sparse_terms: &[SparseSupportProduct<F>],
    transcript: &mut IOPTranscript<F>,
) -> Result<IOPProof<F>, PolyIOPErrors> {
    let start = start_timer!(|| "mixed dense+sparse sum check prove");

    if dense_poly.aux_info.num_variables == 0 {
        return Err(PolyIOPErrors::InvalidParameters(
            "Attempt to prove a constant.".to_string(),
        ));
    }

    for sparse_term in sparse_terms {
        if sparse_term.num_vars() != dense_poly.aux_info.num_variables {
            return Err(PolyIOPErrors::InvalidParameters(format!(
                "sparse term num_vars mismatch: {} vs {}",
                sparse_term.num_vars(),
                dense_poly.aux_info.num_variables
            )));
        }
    }

    let aux_info = VPAuxInfo::<F> {
        max_degree: max(dense_poly.aux_info.max_degree, 2),
        num_variables: dense_poly.aux_info.num_variables,
        phantom: PhantomData,
    };
    transcript.append_serializable_element(b"aux info", &aux_info)?;

    let mut dense_state = IOPProverState::prover_init(dense_poly)?;
    let mut sparse_states = sparse_terms.to_vec();
    let mut challenge = None;
    let mut prover_msgs = Vec::with_capacity(aux_info.num_variables);
    for _ in 0..aux_info.num_variables {
        if let Some(challenge_value) = challenge {
            for sparse_state in sparse_states.iter_mut() {
                sparse_state.apply_challenge(challenge_value)?;
            }
        }

        let mut prover_msg =
            IOPProverState::prove_round_and_update_state(&mut dense_state, &challenge)?;
        if prover_msg.evaluations.len() < aux_info.max_degree + 1 {
            prover_msg
                .evaluations
                .resize(aux_info.max_degree + 1, F::zero());
        }
        for sparse_state in sparse_states.iter() {
            let evaluations = sparse_state.round_evaluations(aux_info.max_degree);
            for (target, sparse_value) in prover_msg.evaluations.iter_mut().zip(evaluations) {
                *target += sparse_value;
            }
        }

        transcript.append_serializable_element(b"prover msg", &prover_msg)?;
        prover_msgs.push(prover_msg);
        challenge = Some(transcript.get_and_append_challenge(b"Internal round")?);
    }

    if let Some(point) = challenge {
        dense_state.challenges.push(point);
    }

    end_timer!(start);
    Ok(IOPProof {
        point: dense_state.challenges,
        proofs: prover_msgs,
    })
}

fn normalize_sparse_entries<F: PrimeField>(
    entries: impl IntoIterator<Item = (usize, F)>,
    domain_size: usize,
) -> Result<Vec<SparseEntry<F>>, PolyIOPErrors> {
    let mut normalized = entries
        .into_iter()
        .filter_map(|(index, value)| (!value.is_zero()).then_some((index, value)))
        .collect::<Vec<_>>();
    normalized.sort_by_key(|(index, _)| *index);

    let mut deduped: Vec<SparseEntry<F>> = Vec::with_capacity(normalized.len());
    for (index, value) in normalized {
        if index >= domain_size {
            return Err(PolyIOPErrors::InvalidParameters(format!(
                "sparse support index {} exceeds domain size {}",
                index, domain_size
            )));
        }
        if let Some(last) = deduped.last_mut() {
            if last.index == index {
                last.value += value;
                if last.value.is_zero() {
                    deduped.pop();
                }
                continue;
            }
        }
        deduped.push(SparseEntry { index, value });
    }

    Ok(deduped)
}

fn build_sparse_pairs<F: PrimeField>(entries: &[SparseEntry<F>]) -> Vec<SparsePair<F>> {
    let mut pairs = Vec::new();
    let mut cursor = 0usize;
    while cursor < entries.len() {
        let parent_index = entries[cursor].index >> 1;
        let mut left = F::zero();
        let mut right = F::zero();
        while cursor < entries.len() && (entries[cursor].index >> 1) == parent_index {
            if entries[cursor].index & 1 == 0 {
                left += entries[cursor].value;
            } else {
                right += entries[cursor].value;
            }
            cursor += 1;
        }
        if !left.is_zero() || !right.is_zero() {
            pairs.push(SparsePair {
                parent_index,
                left,
                right,
            });
        }
    }
    pairs
}

fn collapse_pairs<F: PrimeField>(pairs: Vec<SparsePair<F>>, challenge: F) -> Vec<SparseEntry<F>> {
    let mut collapsed = Vec::with_capacity(pairs.len());
    for pair in pairs {
        let value = pair.left + (pair.right - pair.left) * challenge;
        if !value.is_zero() {
            collapsed.push(SparseEntry {
                index: pair.parent_index,
                value,
            });
        }
    }
    collapsed
}

#[cfg(test)]
mod tests {
    use super::{prove_mixed_sparse_products, SparseSupportProduct};
    use crate::poly_iop::prelude::SumCheck;
    use crate::poly_iop::PolyIOP;
    use arithmetic::VirtualPolynomial;
    use ark_bls12_381::Fr;
    use ark_ff::{One, Zero};
    use ark_poly::{DenseMultilinearExtension, Polynomial};
    use std::sync::Arc;
    use transcript::IOPTranscript;

    fn dense_mle(num_vars: usize, entries: &[(usize, Fr)]) -> Arc<DenseMultilinearExtension<Fr>> {
        let mut evaluations = vec![Fr::zero(); 1usize << num_vars];
        for (index, value) in entries {
            evaluations[*index] = *value;
        }
        Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            evaluations,
        ))
    }

    #[test]
    fn mixed_sparse_product_matches_dense_sumcheck_proof() {
        let num_vars = 3usize;
        let left_entries = vec![(1usize, Fr::from(2u64)), (4usize, Fr::from(5u64))];
        let right_entries = vec![(1usize, Fr::from(7u64)), (4usize, Fr::from(11u64))];
        let sparse_term = SparseSupportProduct::new(
            num_vars,
            -Fr::one(),
            left_entries.clone(),
            right_entries.clone(),
        )
        .unwrap();

        let zero_dense = VirtualPolynomial::<Fr>::new(num_vars);
        let dense_left = dense_mle(num_vars, &left_entries);
        let dense_right = dense_mle(num_vars, &right_entries);
        let mut dense_poly = VirtualPolynomial::<Fr>::new(num_vars);
        dense_poly
            .add_mle_list([dense_left, dense_right], -Fr::one())
            .unwrap();

        let mut mixed_transcript = IOPTranscript::<Fr>::new(b"mixed");
        let mixed_proof =
            prove_mixed_sparse_products(&zero_dense, &[sparse_term], &mut mixed_transcript)
                .unwrap();

        let mut dense_transcript = IOPTranscript::<Fr>::new(b"mixed");
        let dense_proof = PolyIOP::prove(&dense_poly, &mut dense_transcript).unwrap();

        assert_eq!(mixed_proof, dense_proof);
    }

    #[test]
    fn mixed_dense_and_two_sparse_products_match_dense_sumcheck_proof() {
        let num_vars = 3usize;
        let dense_factor_l = dense_mle(
            num_vars,
            &[
                (0usize, Fr::from(3u64)),
                (2usize, Fr::from(5u64)),
                (7usize, Fr::from(9u64)),
            ],
        );
        let dense_factor_r = dense_mle(
            num_vars,
            &[
                (1usize, Fr::from(4u64)),
                (2usize, Fr::from(6u64)),
                (6usize, Fr::from(8u64)),
            ],
        );
        let sparse_term_a = SparseSupportProduct::new(
            num_vars,
            -Fr::from(5u64),
            vec![(1usize, Fr::from(2u64)), (4usize, Fr::from(7u64))],
            vec![(1usize, Fr::from(11u64)), (4usize, Fr::from(13u64))],
        )
        .unwrap();
        let sparse_term_b = SparseSupportProduct::new(
            num_vars,
            Fr::from(3u64),
            vec![(0usize, Fr::from(17u64)), (5usize, Fr::from(19u64))],
            vec![(0usize, Fr::from(23u64)), (5usize, Fr::from(29u64))],
        )
        .unwrap();

        let mut dense_poly = VirtualPolynomial::<Fr>::new(num_vars);
        dense_poly
            .add_mle_list(
                [dense_factor_l.clone(), dense_factor_r.clone()],
                Fr::from(2u64),
            )
            .unwrap();
        dense_poly
            .add_mle_list(
                [
                    dense_mle(
                        num_vars,
                        &[(1usize, Fr::from(2u64)), (4usize, Fr::from(7u64))],
                    ),
                    dense_mle(
                        num_vars,
                        &[(1usize, Fr::from(11u64)), (4usize, Fr::from(13u64))],
                    ),
                ],
                -Fr::from(5u64),
            )
            .unwrap();
        dense_poly
            .add_mle_list(
                [
                    dense_mle(
                        num_vars,
                        &[(0usize, Fr::from(17u64)), (5usize, Fr::from(19u64))],
                    ),
                    dense_mle(
                        num_vars,
                        &[(0usize, Fr::from(23u64)), (5usize, Fr::from(29u64))],
                    ),
                ],
                Fr::from(3u64),
            )
            .unwrap();

        let mut mixed_dense_component = VirtualPolynomial::<Fr>::new(num_vars);
        mixed_dense_component
            .add_mle_list([dense_factor_l, dense_factor_r], Fr::from(2u64))
            .unwrap();

        let mut mixed_transcript = IOPTranscript::<Fr>::new(b"mixed-dense+sparse");
        let mixed_proof = prove_mixed_sparse_products(
            &mixed_dense_component,
            &[sparse_term_a, sparse_term_b],
            &mut mixed_transcript,
        )
        .unwrap();

        let mut dense_transcript = IOPTranscript::<Fr>::new(b"mixed-dense+sparse");
        let dense_proof = PolyIOP::prove(&dense_poly, &mut dense_transcript).unwrap();

        assert_eq!(mixed_proof, dense_proof);
    }

    #[test]
    fn sparse_support_product_evaluates_factors_exactly() {
        let num_vars = 2usize;
        let sparse_term = SparseSupportProduct::new(
            num_vars,
            Fr::one(),
            vec![(0usize, Fr::from(3u64)), (3usize, Fr::from(4u64))],
            vec![(1usize, Fr::from(5u64)), (3usize, Fr::from(6u64))],
        )
        .unwrap();
        let point = [Fr::from(2u64), Fr::from(3u64)];
        let (left, right) = sparse_term.evaluate_factors_at_point(&point).unwrap();

        let dense_left = dense_mle(
            num_vars,
            &[(0usize, Fr::from(3u64)), (3usize, Fr::from(4u64))],
        );
        let dense_right = dense_mle(
            num_vars,
            &[(1usize, Fr::from(5u64)), (3usize, Fr::from(6u64))],
        );

        assert_eq!(left, dense_left.evaluate(&point.to_vec()));
        assert_eq!(right, dense_right.evaluate(&point.to_vec()));
    }
}
