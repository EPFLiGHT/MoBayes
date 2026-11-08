import numpy as np
from typing import Dict, List

class DiseaseTaxonomy:
    """
    Manages the hierarchical structure of diseases.
    Enables marginalizing probabilities up the tree.
    """
    def __init__(self, leaf_diseases: List[str], parent_mapping: Dict[str, List[str]]):
        """
        Args:
            leaf_diseases: Ordered list of atomic diseases matching the BeliefState vector.
            parent_mapping: Dict of Category -> List of child Categories or Leaves.
                            e.g., {'Respiratory': ['Flu', 'COVID-19']}
        """
        self.leaf_diseases = leaf_diseases
        self.disease_to_idx = {d: i for i, d in enumerate(leaf_diseases)}
        self.mapping = parent_mapping
        
        # Determine root nodes (nodes that are never children)
        all_children = set()
        for children in parent_mapping.values():
            all_children.update(children)
        self.roots = [node for node in parent_mapping.keys() if node not in all_children]

    def get_leaves_for_node(self, node: str) -> List[str]:
        """Recursively gets all leaf diseases under a specific node."""
        # If it's a leaf, return it
        if node in self.disease_to_idx:
            return [node]
        # Otherwise, gather leaves from children
        leaves = []
        for child in self.mapping.get(node, []):
            leaves.extend(self.get_leaves_for_node(child))
        return leaves

    def get_leaf_indices(self, node: str) -> List[int]:
        leaves = self.get_leaves_for_node(node)
        return [self.disease_to_idx[l] for l in leaves if l in self.disease_to_idx]

    def marginalize(self, posterior: np.ndarray, node: str) -> float:
        """Computes the sum probability for an entire category."""
        indices = self.get_leaf_indices(node)
        if not indices: return 0.0
        return np.sum(posterior[indices])

    def get_category_distribution(self, posterior: np.ndarray, categories: List[str]) -> np.ndarray:
        """Returns a probability distribution over a specific set of disjoint categories."""
        dist = np.array([self.marginalize(posterior, cat) for cat in categories])
        # Ensure it normalizes if the categories don't perfectly cover 100% of leaves
        total = np.sum(dist)
        if total > 0: dist /= total
        return dist
