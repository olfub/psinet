# copied parts of the code from causalscbench.evaluation.statistical_evaluation
# for that, see the following copyright
"""
Copyright (C) 2022  GlaxoSmithKline plc - Mathieu Chevalley;

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
from typing import Dict, List
import time

import numpy as np
import scipy
import torch


class Evaluator(object):
    def __init__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        p_value_threshold=0.05,
        num_threads_pytorch=20
    ) -> None:
        """
        Evaluation module to quantitatively evaluate a model using held-out data.

        Args:
            expression_matrix: a numpy matrix of expression data of size [nb_samples, nb_genes]
            interventions: a list of size [nb_samples] that indicates which gene has been perturb. "non-targeting" means no gene has been perturbed (observational data)
            gene_names: name of the genes in the expression matrix
            p_value_threshold: threshold for statistical significance, default 0.05
        """
        self.gene_to_index = dict(zip(gene_names, range(len(gene_names))))
        self.gene_to_interventions = dict()
        for i, intervention in enumerate(interventions):
            self.gene_to_interventions.setdefault(intervention, []).append(i)
        self.expression_matrix = expression_matrix
        self.p_value_threshold = p_value_threshold
        self.gene_names = gene_names
        self.output_directory = None
        torch.set_num_threads(num_threads_pytorch)

    def get_observational(self, child: str) -> np.array:
        """
        Return all the samples for gene "child" in cells where there was no perturbations

        Args:
            child: Gene name of child to get samples for

        Returns:
            np.array matrix of corresponding samples
        """
        return self.get_interventional(child, "non-targeting")

    def get_interventional(self, child: str, parent: str) -> np.array:
        """
        Return all the samples for gene "child" in cells where "parent" was perturbed

        Args:
            child: Gene name of child to get samples for
            parent: Gene name of gene that must have been perturbed

        Returns:
            np.array matrix of corresponding samples
        """
        return self.expression_matrix[
            self.gene_to_interventions[parent], self.gene_to_index[child]
        ]

    def set_output_directory(self, directory: str) -> None:
        self.output_directory = directory
    
    def evaluate_model_distance(self, model, model_name="einet") -> Dict:
        device = next(model.parameters()).device

        l1_dist = 0
        l2_dist = 0
        count = 0
        # loop for interventions
        settings = ["non-targeting"] + self.gene_names
        # if len(self.gene_names) > 828:
        #     # i.e. weissmann_k562 dataset
        #     settings = list(np.array(settings)[[0, 828, 8, 613, 305, 571]])
        # else:
        #     # i.e. weissmann_rpe1 dataset
        #     settings = list(np.array(settings)[[0, 412, 8, 613, 305, 571]])
        start_time = time.time()
        for gene_name_interv in settings:
            # the variables to compare to

            # intervention vector
            train_interventions = torch.zeros((1, len(self.gene_names))).to(device)
            if gene_name_interv != "non-targeting":
                train_interventions[:, self.gene_names.index(gene_name_interv)] = 1

            if model_name == "einet":
                mpe = model.mpe(train_interventions)
            elif model_name == "iSPN":
                mpe = model.predict(train_interventions, torch.zeros((1, len(self.gene_names))).to(device), torch.ones((1, len(self.gene_names))).to(device))
            else:
                # assuming this must be NCM
                if gene_name_interv == "non-targeting":
                    ncm_samples = model.sampling(num_samples=1024)
                else:
                    ncm_samples = model.sampling(num_samples=1024, do={"Instr_"+gene_name_interv: 1})
                ncm_samples = {gene: ncm_samples[gene].cpu().detach().numpy() for gene in ncm_samples if "Instr_" not in gene}

            for gene_name_var in self.gene_names:
                # get test samples
                test_samples = self.get_interventional(gene_name_var, gene_name_interv)
                samples_mean = test_samples.mean()
                if model_name == "einet" or model_name == "iSPN":
                    l1_dist += abs(mpe[0,self.gene_names.index(gene_name_var)] - samples_mean).item()
                    l2_dist += (mpe[0,self.gene_names.index(gene_name_var)] - samples_mean).item()**2
                else:
                    l1_dist += abs(ncm_samples[gene_name_var].mean() - samples_mean)
                    l2_dist += (ncm_samples[gene_name_var].mean() - samples_mean)**2
                # print progress
                count += 1
                if count % 100 == 0:
                    print(f"L1/L2 Progress: {100*(count)/len(self.gene_names)**2}%, l1 avg: {l1_dist / count}, l2 avg: {l2_dist / count}", end="\r")
        eval_time = time.time() - start_time
        with open(f"{self.output_directory}/l1l2_eval_time.txt", "w") as logging:
            logging.write(f"Eval time:\n{str(eval_time)}\n")
        return {"l1_avg" : l1_dist / count, "l2_avg" : l2_dist / count}
