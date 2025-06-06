############################################
#               STEP 2: apriori_core.py 
# Apriori Algorithm Implementation for Association Rules Mining
# This file implements the core Apriori algorithm to find frequent itemsets
# and generate association rules with different support thresholds
# EDUARDO JUNQUEIRA 
# TOPIC 1 ASSOCIATION RULES WITH @APRIORI ALGORITHM PROJECT Data Mining
############################################

############################################
# Import libraries
############################################
import time  # For measuring execution time
from collections import defaultdict, Counter  # For counting items and storing sets
from itertools import combinations  # For generating itemset combinations
from data_loader import load_online_retail_transactions  # Import  Step 1 function file data_loader


############################################
# Class: AprioriAlgorithm
# Core implementation of the Apriori algorithm
############################################

class AprioriAlgorithm:
    def __init__(self, verbose=True):
        """
        Initialize Apriori Algorithm
        
        Args:
            verbose (bool): Print processing information
        """
        self.verbose = verbose  # Controls print output for progress/debug
        self.transactions = []  # Stores the list of transactions (lists of items)
        self.frequent_itemsets = {}  # Dictionary for frequent itemsets by size
        self.all_frequent_itemsets = []  # Flat list of all frequent itemsets
        
    def load_transactions(self, limit=20):
        """
        Load transactions using our data_loader from Step 1
        
        Args:
            limit (int): Number of transactions to load
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"LOADING TRANSACTIONS (LIMIT: {limit} for performance)")
            print(f"{'='*60}")
            
        self.transactions = load_online_retail_transactions(limit=limit, verbose=False)  # Load transactions from data_loader
        
        filtered_transactions = []
        for transaction in self.transactions:
            if len(transaction) <= 10:  # Only keep transactions with 10 or fewer items
                 filtered_transactions.append(list(set(transaction)))  # Remove duplicates from each transaction
        
        self.transactions = [set(trans) for trans in filtered_transactions]  # Convert to sets for faster operations
        
        if self.verbose:
            print(f"Loaded and filtered to {len(self.transactions)} manageable transactions")
            
            # Show sample transactions
            print(f"\nSample transactions:")
            for i, trans in enumerate(self.transactions[:5]):
                print(f"  Transaction {i+1} ({len(trans)} items): {sorted(list(trans))}")
        
        return len(self.transactions)  # Return the number of loaded transactions
    
    def get_item_support(self, itemset):
        """
        Calculate support for a given itemset
        
        Args:
            itemset (set): Set of items
            
        Returns:
            float: Support value (0.0 to 1.0)
        """
        if not self.transactions:
            return 0.0  # Return zero if no transactions
            
        count = 0  # Counter for itemset appearances
        itemset_set = set(itemset) if not isinstance(itemset, set) else itemset  # Ensure itemset is a set
        
        for transaction in self.transactions:
            if itemset_set.issubset(set(transaction)):  # Check if itemset is in transaction
                count += 1
        
        return count / len(self.transactions)  # Calculate and return support
    
    def get_frequent_1_itemsets(self, min_support):
        """
        Find frequent 1-itemsets (single items)
        
        Args:
            min_support (float): Minimum support threshold
            
        Returns:
            list: List of frequent 1-itemsets
        """
        if self.verbose:
            print(f"\nFinding frequent 1-itemsets (min_support: {min_support})")
            
        # Count all items in all transactions
        item_counts = Counter()
        for transaction in self.transactions:
            for item in transaction:
                item_counts[item] += 1
        
        total_transactions = len(self.transactions)
        min_count = int(min_support * total_transactions)  # Minimum count for support
        
        frequent_items = []
        for item, count in item_counts.items():
            if count >= min_count:  # Check if item meets support
                frequent_items.append(frozenset([item]))  # Store as frozenset for set operations
        
        if self.verbose:
            print(f"Found {len(frequent_items)} frequent 1-itemsets")
            for i, itemset in enumerate(frequent_items):
                support = self.get_item_support(itemset)
                print(f"  {i+1}. {sorted(list(itemset))} (support: {support:.4f})")
            
        return frequent_items  # Return list of frequent 1-itemsets
    
    def generate_candidates(self, frequent_itemsets, k):
        """
        Generate candidate k-itemsets from frequent (k-1)-itemsets
        
        Args:
            frequent_itemsets (list): List of frequent (k-1)-itemsets
            k (int): Size of itemsets to generate
            
        Returns:
            list: List of candidate k-itemsets
        """
        # FIX: More efficient candidate generation
        candidates = []
        items = sorted({item for itemset in frequent_itemsets for item in itemset})  # Unique items from all itemsets
        
        # Generate candidates using combinations for k=2
        if k == 2:
            candidates = [frozenset([i, j]) for i in items for j in items if i < j]
        else:
            # For k > 2, use lexicographic ordering approach
            frequent_list = [sorted(list(itemset)) for itemset in frequent_itemsets]
            frequent_list.sort()
            
            for i in range(len(frequent_list)):
                for j in range(i + 1, len(frequent_list)):
                    # Check if first k-2 items are the same
                    if frequent_list[i][:-1] == frequent_list[j][:-1]:
                        candidate = frozenset(frequent_list[i] + [frequent_list[j][-1]])
                        if len(candidate) == k:
                            candidates.append(candidate)
        
        return candidates  # Return list of candidate itemsets
    
    def prune_candidates(self, candidates, frequent_itemsets):
        """
        Prune candidates that have infrequent subsets
        
        Args:
            candidates (list): List of candidate itemsets
            frequent_itemsets (list): List of frequent itemsets from previous iteration
            
        Returns:
            list: Pruned list of candidates
        """
        
        # FIX: Use set for faster lookups
        freq_set = set(frequent_itemsets)  # Convert to set for O(1) lookups
        pruned = []
        for candidate in candidates:
            subsets = combinations(candidate, len(candidate)-1)  # All (k-1)-subsets
            if all(frozenset(sub) in freq_set for sub in subsets):  # Keep only if all subsets are frequent
                pruned.append(candidate)
        return pruned  # Return pruned candidates
    
    def find_frequent_itemsets(self, min_support):
        """
        Main Apriori algorithm to find all frequent itemsets
        
        Args:
            min_support (float): Minimum support threshold
            
        Returns:
            dict: Dictionary with k as key and list of frequent k-itemsets as value
        """
        if not self.transactions:
            print("Error: No transactions loaded!")
            return {}
            
        start_time = time.time()  # Start timer
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"APRIORI ALGORITHM - FINDING FREQUENT ITEMSETS")
            print(f"{'='*80}")
            print(f"Minimum support threshold: {min_support}")
            print(f"Total transactions: {len(self.transactions)}")
        
        # Step 1: Find frequent 1-itemsets
        frequent_itemsets = {1: self.get_frequent_1_itemsets(min_support)}
        
        if not frequent_itemsets[1]:
            if self.verbose:
                print("No frequent 1-itemsets found!")
            return frequent_itemsets
        
        k = 2
        while frequent_itemsets[k-1] and k <= 4:  # LIMIT TO 4-ITEMSETS FOR PERFORMANCE
            if self.verbose:
                print(f"\n{'-'*40}")
                print(f"FINDING FREQUENT {k}-ITEMSETS")
                print(f"{'-'*40}")
            
            # Generate candidates
            candidates = self.generate_candidates(frequent_itemsets[k-1], k)
            
            if not candidates:
                if self.verbose:
                    print(f"No candidates generated for {k}-itemsets")
                break
            
            # Prune candidates
            pruned_candidates = self.prune_candidates(candidates, frequent_itemsets[k-1])
            
            if self.verbose:
                print(f"Generated {len(candidates)} candidates")
                print(f"After pruning: {len(pruned_candidates)} candidates")
            
            # Find frequent itemsets among candidates
            frequent_k_itemsets = []
            for candidate in pruned_candidates:
                support = self.get_item_support(candidate)
                if support >= min_support:
                    frequent_k_itemsets.append(candidate)
            
            if self.verbose:
                print(f"Frequent {k}-itemsets found: {len(frequent_k_itemsets)}")
                if len(frequent_k_itemsets) <= 10:  # Show first 10 to avoid clutter
                    for i, itemset in enumerate(frequent_k_itemsets):
                        support = self.get_item_support(itemset)
                        items = sorted(list(itemset))
                        print(f"  {i+1}. {items} (support: {support:.4f})")
            
            if frequent_k_itemsets:
                frequent_itemsets[k] = frequent_k_itemsets  # Store frequent itemsets of size k
                k += 1
            else:
                break  # Stop if no frequent itemsets found
        
        # Store results
        self.frequent_itemsets = frequent_itemsets  # Store in object
        
        # Create flat list of all frequent itemsets
        self.all_frequent_itemsets = []
        for k_itemsets in frequent_itemsets.values():
            self.all_frequent_itemsets.extend(k_itemsets)
        
        end_time = time.time()
        processing_time = end_time - start_time  # Calculate processing time
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"APRIORI ALGORITHM RESULTS")
            print(f"{'='*60}")
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"Total frequent itemsets found: {len(self.all_frequent_itemsets)}")
            
            for k, itemsets in frequent_itemsets.items():
                print(f"  Frequent {k}-itemsets: {len(itemsets)}")
        
        return frequent_itemsets  # Return dictionary of frequent itemsets
    
    def run_quick_test(self):
        """
        Quick test function to verify the algorithm works
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"RUNNING QUICK TEST")
            print(f"{'='*60}")
        
        # Load small dataset
        self.load_transactions(limit=10)
        
        # Test with lower support threshold for real data
        results = self.find_frequent_itemsets(0.05)
        
        return results  # Return frequent itemsets from quick test
    
    def run_multiple_support_thresholds(self, support_thresholds=[0.1, 0.05, 0.02]):  # HIGHER THRESHOLDS FOR TESTING
        """
        Run Apriori algorithm with multiple support thresholds as shown in flowchart
        
        Args:
            support_thresholds (list): List of support thresholds to test
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"RUNNING APRIORI WITH MULTIPLE SUPPORT THRESHOLDS")
            print(f"{'='*80}")
        
        results = {}
        
        for threshold in support_thresholds:
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"TESTING MINIMUM SUPPORT: {threshold}")
                if threshold == 0.005:
                    print("(LENIENT THRESHOLD)")
                elif threshold == 0.01:
                    print("(MODERATE THRESHOLD)")
                elif threshold == 0.02:
                    print("(STRICTER THRESHOLD)")
                print(f"{'='*80}")
            
            frequent_itemsets = self.find_frequent_itemsets(threshold)
            results[threshold] = frequent_itemsets  # Store results for each threshold
            
            # Count total itemsets for this threshold
            total_itemsets = sum(len(itemsets) for itemsets in frequent_itemsets.values())
            
            if self.verbose:
                print(f"\nSUMMARY for support threshold {threshold}:")
                print(f"Total frequent itemsets: {total_itemsets}")
        
        return results  # Return dictionary of results for all thresholds

############################################
# Main execution function
############################################

def main():
    """
    Main function to run Step 2 of the Apriori algorithm
    """
    print("="*80)
    print("@APRIORI ALGORITHM PROJECT - ASSOCIATION RULES MINING")
    print("EDUARDO JUNQUEIRA - @EDAMI")
    print("="*80)
    print("STEP 2: APRIORI CORE ALGORITHM - FINDING FREQUENT ITEMSETS")
    print("="*80)
    
    # Initialize Apriori algorithm
    apriori = AprioriAlgorithm(verbose=True)

    # Load transactions from Step 1
    transaction_count = apriori.load_transactions(limit=50)  # MODERATE LIMIT FOR REAL DATA
    
    if transaction_count == 0:
        print("Error: No transactions loaded. Please check Step 1 (data_loader.py)")
        return
    
    # Run algorithm with multiple support thresholds (as per flowchart)
    support_thresholds = [0.1, 0.05]  # LOWER THRESHOLDS FOR REAL DATA
    results = apriori.run_multiple_support_thresholds(support_thresholds)
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY - STEP 2 COMPLETED")
    print(f"{'='*80}")
    
    for threshold, frequent_itemsets in results.items():
        total_itemsets = sum(len(itemsets) for itemsets in frequent_itemsets.values())
        print(f"Support {threshold}: {total_itemsets} frequent itemsets")
    
    print(f"\nStep 2 completed successfully!")
    print(f"Ready for Step 3: Generate association rules")


############################################
# Execute main function if script is run directly
############################################

if __name__ == "__main__":
    main()
