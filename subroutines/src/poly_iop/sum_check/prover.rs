// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Prover subroutines for a SumCheck protocol.

use super::SumCheckProver;
use crate::poly_iop::{
    errors::PolyIOPErrors,
    structs::{IOPProverMessage, IOPProverState},
};
use arithmetic::VirtualPolynomial;
use ark_ff::{batch_inversion, PrimeField};
use ark_poly::DenseMultilinearExtension;
use ark_std::{cfg_into_iter, end_timer, start_timer, vec::Vec};
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

impl<F: PrimeField> SumCheckProver<F> for IOPProverState<F> {
    type VirtualPolynomial = VirtualPolynomial<F>;
    type ProverMessage = IOPProverMessage<F>;

    /// Initialize the prover state to argue for the sum of the input polynomial
    /// over {0,1}^`num_vars`.
    fn prover_init(polynomial: &Self::VirtualPolynomial) -> Result<Self, PolyIOPErrors> {
        let start = start_timer!(|| "sum check prover init");
        let virtual_poly_build = start_timer!(|| {
            format!(
                "sumcheck_f_virtual_poly_build prover_init num_vars={} products={} flattened_mles={} max_degree={}",
                polynomial.aux_info.num_variables,
                polynomial.products.len(),
                polynomial.flattened_ml_extensions.len(),
                polynomial.aux_info.max_degree
            )
        });
        if polynomial.aux_info.num_variables == 0 {
            return Err(PolyIOPErrors::InvalidParameters(
                "Attempt to prove a constant.".to_string(),
            ));
        }
        let alloc_timer = start_timer!(|| {
            format!(
                "sumcheck_f_alloc prover_state num_rounds={} max_degree={}",
                polynomial.aux_info.num_variables, polynomial.aux_info.max_degree
            )
        });
        let challenges = Vec::with_capacity(polynomial.aux_info.num_variables);
        let extrapolation_aux = (1..polynomial.aux_info.max_degree)
            .map(|degree| {
                let points = (0..1 + degree as u64).map(F::from).collect::<Vec<_>>();
                let weights = barycentric_weights(&points);
                (points, weights)
            })
            .collect();
        end_timer!(alloc_timer);
        let clone_timer = start_timer!(|| {
            format!(
                "sumcheck_f_clone prover_state_virtual_polynomial products={} flattened_mles={}",
                polynomial.products.len(),
                polynomial.flattened_ml_extensions.len()
            )
        });
        let poly = polynomial.clone();
        end_timer!(clone_timer);
        end_timer!(virtual_poly_build);
        end_timer!(start);

        Ok(Self {
            challenges,
            round: 0,
            poly,
            extrapolation_aux,
        })
    }

    /// Receive message from verifier, generate prover message, and proceed to
    /// next round.
    ///
    /// Main algorithm used is from section 3.2 of [XZZPS19](https://eprint.iacr.org/2019/317.pdf#subsection.3.2).
    fn prove_round_and_update_state(
        &mut self,
        challenge: &Option<F>,
    ) -> Result<Self::ProverMessage, PolyIOPErrors> {
        // let start =
        //     start_timer!(|| format!("sum check prove {}-th round and update state",
        // self.round));

        if self.round >= self.poly.aux_info.num_variables {
            return Err(PolyIOPErrors::InvalidProver(
                "Prover is not active".to_string(),
            ));
        }

        // let fix_argument = start_timer!(|| "fix argument");

        // Step 1:
        // fix argument and evaluate f(x) over x_m = r; where r is the challenge
        // for the current round, and m is the round number, indexed from 1
        //
        // i.e.:
        // at round m <= n, for each mle g(x_1, ... x_n) within the flattened_mle
        // which has already been evaluated to
        //
        //    g(r_1, ..., r_{m-1}, x_m ... x_n)
        //
        // eval g over r_m, and mutate g to g(r_1, ... r_m,, x_{m+1}... x_n)
        let round_index = self.round;
        let clone_flattened = start_timer!(|| {
            format!(
                "sumcheck_f_clone round={} flattened_mles={}",
                round_index,
                self.poly.flattened_ml_extensions.len()
            )
        });
        let mut flattened_ml_extensions: Vec<DenseMultilinearExtension<F>> = self
            .poly
            .flattened_ml_extensions
            .iter()
            .map(|x| x.as_ref().clone())
            .collect();
        end_timer!(clone_flattened);

        let bind_timer = start_timer!(|| {
            format!(
                "sumcheck_f_round_bind index={} has_challenge={} flattened_mles={}",
                round_index,
                challenge.is_some(),
                flattened_ml_extensions.len()
            )
        });
        if let Some(chal) = challenge {
            if self.round == 0 {
                return Err(PolyIOPErrors::InvalidProver(
                    "first round should be prover first.".to_string(),
                ));
            }
            self.challenges.push(*chal);

            let r = self.challenges[self.round - 1];
            #[cfg(feature = "parallel")]
            flattened_ml_extensions
                .par_iter_mut()
                .for_each(|mle| bind_first_variable_in_place(mle, r));
            #[cfg(not(feature = "parallel"))]
            flattened_ml_extensions
                .iter_mut()
                .for_each(|mle| bind_first_variable_in_place(mle, r));
        } else if self.round > 0 {
            return Err(PolyIOPErrors::InvalidProver(
                "verifier message is empty".to_string(),
            ));
        }
        // end_timer!(fix_argument);
        end_timer!(bind_timer);

        self.round += 1;

        let products_clone = start_timer!(|| {
            format!(
                "sumcheck_f_clone round={} products={}",
                round_index,
                self.poly.products.len()
            )
        });
        let products_list = self.poly.products.clone();
        end_timer!(products_clone);
        let alloc_timer = start_timer!(|| {
            format!(
                "sumcheck_f_alloc round={} products_sum_len={}",
                round_index,
                self.poly.aux_info.max_degree + 1
            )
        });
        let mut products_sum = vec![F::zero(); self.poly.aux_info.max_degree + 1];
        end_timer!(alloc_timer);

        // Step 2: generate sum for the partial evaluated polynomial:
        // f(r_1, ... r_m,, x_{m+1}... x_n)

        let round_eval = start_timer!(|| {
            format!(
                "sumcheck_f_round_eval index={} remaining_vars={} products={} max_degree={}",
                round_index,
                self.poly.aux_info.num_variables - self.round,
                products_list.len(),
                self.poly.aux_info.max_degree
            )
        });
        let remaining_rows = 1usize << (self.poly.aux_info.num_variables - self.round);
        products_list.iter().for_each(|(coefficient, products)| {
            let dense_scan = start_timer!(|| {
                format!(
                    "sumcheck_f_round_dense_scan index={} multiplicands={} rows={}",
                    round_index,
                    products.len(),
                    remaining_rows
                )
            });
            let mut sum = match products.as_slice() {
                [f] => scan_single_multiplicand(
                    flattened_ml_extensions[*f].evaluations.as_slice(),
                    remaining_rows,
                )
                .to_vec(),
                [f0, f1] => scan_pair_multiplicands(
                    flattened_ml_extensions[*f0].evaluations.as_slice(),
                    flattened_ml_extensions[*f1].evaluations.as_slice(),
                    remaining_rows,
                )
                .to_vec(),
                _ => {
                    let product_tables = products
                        .iter()
                        .map(|f| flattened_ml_extensions[*f].evaluations.as_slice())
                        .collect::<Vec<_>>();
                    scan_product_generic(&product_tables, remaining_rows)
                },
            };
            end_timer!(dense_scan);
            sum.iter_mut().for_each(|sum| *sum *= coefficient);
            let extraploation = if self.poly.aux_info.max_degree > products.len() {
                let extrapolate_timer = start_timer!(|| {
                    format!(
                        "sumcheck_f_round_extrapolate index={} multiplicands={} extrapolated_points={}",
                        round_index,
                        products.len(),
                        self.poly.aux_info.max_degree - products.len()
                    )
                });
                let values =
                    cfg_into_iter!(0..self.poly.aux_info.max_degree - products.len())
                        .map(|i| {
                            let (points, weights) = &self.extrapolation_aux[products.len() - 1];
                            let at = F::from((products.len() + 1 + i) as u64);
                            extrapolate(points, weights, &sum, &at)
                        })
                        .collect::<Vec<_>>();
                end_timer!(extrapolate_timer);
                values
            } else {
                Vec::new()
            };
            products_sum
                .iter_mut()
                .zip(sum.iter().chain(extraploation.iter()))
                .for_each(|(products_sum, sum)| *products_sum += sum);
        });
        end_timer!(round_eval);

        // update prover's state to the partial evaluated polynomial
        let clone_rebind = start_timer!(|| {
            format!(
                "sumcheck_f_clone round={} rebound_flattened_mles={}",
                round_index,
                flattened_ml_extensions.len()
            )
        });
        self.poly.flattened_ml_extensions = flattened_ml_extensions
            .iter()
            .map(|x| Arc::new(x.clone()))
            .collect();
        end_timer!(clone_rebind);

        Ok(IOPProverMessage {
            evaluations: products_sum,
        })
    }
}

fn barycentric_weights<F: PrimeField>(points: &[F]) -> Vec<F> {
    let mut weights = points
        .iter()
        .enumerate()
        .map(|(j, point_j)| {
            points
                .iter()
                .enumerate()
                .filter(|&(i, _point_i)| (i != j))
                .map(|(_i, point_i)| *point_j - point_i)
                .reduce(|acc, value| acc * value)
                .unwrap_or_else(F::one)
        })
        .collect::<Vec<_>>();
    batch_inversion(&mut weights);
    weights
}

fn bind_first_variable_in_place<F: PrimeField>(
    mle: &mut DenseMultilinearExtension<F>,
    challenge: F,
) {
    if mle.num_vars == 0 {
        return;
    }

    let next_len = 1usize << (mle.num_vars - 1);
    for b in 0..next_len {
        let left = mle.evaluations[b << 1];
        let right = mle.evaluations[(b << 1) + 1];
        mle.evaluations[b] = left + challenge * (right - left);
    }
    mle.evaluations.truncate(next_len);
    mle.num_vars -= 1;
}

fn scan_single_multiplicand<F: PrimeField>(table: &[F], remaining_rows: usize) -> [F; 2] {
    #[cfg(feature = "parallel")]
    {
        return cfg_into_iter!(0..remaining_rows)
            .fold(
                || [F::zero(); 2],
                |mut acc, b| {
                    let idx = b << 1;
                    acc[0] += table[idx];
                    acc[1] += table[idx + 1];
                    acc
                },
            )
            .reduce(
                || [F::zero(); 2],
                |mut acc, partial| {
                    acc[0] += partial[0];
                    acc[1] += partial[1];
                    acc
                },
            );
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut acc = [F::zero(); 2];
        for b in 0..remaining_rows {
            let idx = b << 1;
            acc[0] += table[idx];
            acc[1] += table[idx + 1];
        }
        acc
    }
}

fn scan_pair_multiplicands<F: PrimeField>(
    left_table: &[F],
    right_table: &[F],
    remaining_rows: usize,
) -> [F; 3] {
    #[cfg(feature = "parallel")]
    {
        return cfg_into_iter!(0..remaining_rows)
            .fold(
                || [F::zero(); 3],
                |mut acc, b| {
                    let idx = b << 1;

                    let left_0 = left_table[idx];
                    let left_1 = left_table[idx + 1];
                    let left_step = left_1 - left_0;

                    let right_0 = right_table[idx];
                    let right_1 = right_table[idx + 1];
                    let right_step = right_1 - right_0;

                    acc[0] += left_0 * right_0;
                    acc[1] += left_1 * right_1;
                    acc[2] += (left_1 + left_step) * (right_1 + right_step);
                    acc
                },
            )
            .reduce(
                || [F::zero(); 3],
                |mut acc, partial| {
                    acc[0] += partial[0];
                    acc[1] += partial[1];
                    acc[2] += partial[2];
                    acc
                },
            );
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut acc = [F::zero(); 3];
        for b in 0..remaining_rows {
            let idx = b << 1;

            let left_0 = left_table[idx];
            let left_1 = left_table[idx + 1];
            let left_step = left_1 - left_0;

            let right_0 = right_table[idx];
            let right_1 = right_table[idx + 1];
            let right_step = right_1 - right_0;

            acc[0] += left_0 * right_0;
            acc[1] += left_1 * right_1;
            acc[2] += (left_1 + left_step) * (right_1 + right_step);
        }
        acc
    }
}

fn scan_product_generic<F: PrimeField>(product_tables: &[&[F]], remaining_rows: usize) -> Vec<F> {
    #[cfg(feature = "parallel")]
    {
        return cfg_into_iter!(0..remaining_rows)
            .fold(
                || {
                    (
                        vec![F::zero(); product_tables.len()],
                        vec![F::zero(); product_tables.len()],
                        vec![F::zero(); product_tables.len() + 1],
                    )
                },
                |(mut evals, mut steps, mut acc), b| {
                    let idx = b << 1;
                    for ((eval, step), table) in evals
                        .iter_mut()
                        .zip(steps.iter_mut())
                        .zip(product_tables.iter())
                    {
                        let left = table[idx];
                        let right = table[idx + 1];
                        *eval = left;
                        *step = right - left;
                    }

                    acc[0] += evals.iter().copied().product::<F>();
                    for acc_slot in acc[1..].iter_mut() {
                        for (eval, step) in evals.iter_mut().zip(steps.iter()) {
                            *eval += *step;
                        }
                        *acc_slot += evals.iter().copied().product::<F>();
                    }

                    (evals, steps, acc)
                },
            )
            .map(|(_, _, partial)| partial)
            .reduce(
                || vec![F::zero(); product_tables.len() + 1],
                |mut sum, partial| {
                    sum.iter_mut()
                        .zip(partial.iter())
                        .for_each(|(sum, partial)| *sum += partial);
                    sum
                },
            );
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut sum = vec![F::zero(); product_tables.len() + 1];
        let mut evals = vec![F::zero(); product_tables.len()];
        let mut steps = vec![F::zero(); product_tables.len()];

        for b in 0..remaining_rows {
            let idx = b << 1;
            for ((eval, step), table) in evals
                .iter_mut()
                .zip(steps.iter_mut())
                .zip(product_tables.iter())
            {
                let left = table[idx];
                let right = table[idx + 1];
                *eval = left;
                *step = right - left;
            }

            sum[0] += evals.iter().copied().product::<F>();
            for sum_slot in sum[1..].iter_mut() {
                for (eval, step) in evals.iter_mut().zip(steps.iter()) {
                    *eval += *step;
                }
                *sum_slot += evals.iter().copied().product::<F>();
            }
        }

        sum
    }
}

fn extrapolate<F: PrimeField>(points: &[F], weights: &[F], evals: &[F], at: &F) -> F {
    let (coeffs, sum_inv) = {
        let mut coeffs = points.iter().map(|point| *at - point).collect::<Vec<_>>();
        batch_inversion(&mut coeffs);
        coeffs.iter_mut().zip(weights).for_each(|(coeff, weight)| {
            *coeff *= weight;
        });
        let sum_inv = coeffs.iter().sum::<F>().inverse().unwrap_or_default();
        (coeffs, sum_inv)
    };
    coeffs
        .iter()
        .zip(evals)
        .map(|(coeff, eval)| *coeff * eval)
        .sum::<F>()
        * sum_inv
}
