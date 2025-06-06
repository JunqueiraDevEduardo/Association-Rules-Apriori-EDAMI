############################################
#               STEP 4: rule_filter.py 
# Advanced Rule Filtering and Quality Assessment
# This file implements advanced filtering techniques for association rules
# generated in Step 3, applying lift filters and quality metrics
# EDUARDO JUNQUEIRA 
# TOPIC 1 ASSOCIATION RULES WITH @APRIORI ALGORITHM PROJECT Data Mining
############################################

############################################
# Import libraries
############################################
import time  # For measuring execution time
import math  # For mathematical operations
from rules_engine import AssociationRulesGenerator  # Import Step 3 class

############################################
# Class: RuleFilter
# Advanced filtering and quality assessment of association rules
############################################

class RuleFilter:
    def __init__(self, verbose=True):
        """
        Initialize Rule Filter
        
        Args:
            verbose (bool): Print processing information
        """
        self.verbose = verbose
        self.rules_generator = AssociationRulesGenerator(verbose=False)
        self.original_rules = []  # Original rules from Step 3
        self.filtered_rules = []  # Rules after filtering
        self.filter_stats = {}  # Statistics about filtering process
        
    def load_rules_from_step3(self, limit=50, min_support=0.05, min_confidence=0.4):
        """
        Load association rules from Step 3
        
        Args:
            limit (int): Number of transactions to process
            min_support (float): Minimum support threshold
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            bool: Success status
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"LOADING RULES FROM STEP 3")
            print(f"{'='*80}")
            
        # Load frequent itemsets and generate rules
        success = self.rules_generator.load_frequent_itemsets(limit=limit, min_support=min_support)
        
        if not success:
            print("Error: Failed to load frequent itemsets from Step 3!")
            return False
            
        # Generate association rules
        self.original_rules = self.rules_generator.generate_association_rules(min_confidence=min_confidence)
        
        if self.verbose:
            print(f"Loaded {len(self.original_rules)} rules from Step 3")
            
        return True
        
    def filter_by_lift(self, min_lift=1.0, max_lift=float('inf')):
        """
        Filter rules based on lift values
        
        Args:
            min_lift (float): Minimum lift threshold
            max_lift (float): Maximum lift threshold
            
        Returns:
            list: Filtered rules
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"FILTERING BY LIFT: {min_lift} <= lift <= {max_lift}")
            print(f"{'='*60}")
            
        filtered = []
        
        for rule in self.original_rules:
            lift = rule['lift']
            if min_lift <= lift <= max_lift:
                filtered.append(rule)
                
        if self.verbose:
            print(f"Rules before lift filtering: {len(self.original_rules)}")
            print(f"Rules after lift filtering: {len(filtered)}")
            print(f"Filtered out: {len(self.original_rules) - len(filtered)} rules")
            
        return filtered
        
    def filter_by_interest_measures(self, min_conviction=1.0, min_leverage=0.0):
        """
        Filter rules using additional interest measures
        
        Args:
            min_conviction (float): Minimum conviction threshold
            min_leverage (float): Minimum leverage threshold
            
        Returns:
            list: Filtered rules with additional metrics
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"FILTERING BY INTEREST MEASURES")
            print(f"{'='*60}")
            print(f"Minimum conviction: {min_conviction}")
            print(f"Minimum leverage: {min_leverage}")
            
        filtered = []
        
        for rule in self.original_rules.copy():
            # Calculate additional interest measures
            conviction = self.calculate_conviction(rule)
            leverage = self.calculate_leverage(rule)
            jaccard = self.calculate_jaccard(rule)
            
            # Add new metrics to rule
            rule['conviction'] = conviction
            rule['leverage'] = leverage
            rule['jaccard'] = jaccard
            
            # Apply filters
            if conviction >= min_conviction and leverage >= min_leverage:
                filtered.append(rule)
                
        if self.verbose:
            print(f"Rules before interest measure filtering: {len(self.original_rules)}")
            print(f"Rules after interest measure filtering: {len(filtered)}")
            print(f"Filtered out: {len(self.original_rules) - len(filtered)} rules")
            
        return filtered
        
    def calculate_conviction(self, rule):
        """
        Calculate conviction for a rule
        Conviction = (1 - support(consequent)) / (1 - confidence)
        
        Args:
            rule (dict): Association rule
            
        Returns:
            float: Conviction value
        """
        confidence = rule['confidence']
        consequent_support = rule['consequent_support']
        
        if confidence == 1.0:
            return float('inf')  # Perfect confidence
            
        conviction = (1 - consequent_support) / (1 - confidence)
        return conviction
        
    def calculate_leverage(self, rule):
        """
        Calculate leverage for a rule
        Leverage = support(X ∪ Y) - support(X) * support(Y)
        
        Args:
            rule (dict): Association rule
            
        Returns:
            float: Leverage value
        """
        support_union = rule['support']
        support_antecedent = rule['antecedent_support']
        support_consequent = rule['consequent_support']
        
        leverage = support_union - (support_antecedent * support_consequent)
        return leverage
        
    def calculate_jaccard(self, rule):
        """
        Calculate Jaccard coefficient for a rule
        Jaccard = support(X ∪ Y) / (support(X) + support(Y) - support(X ∪ Y))
        
        Args:
            rule (dict): Association rule
            
        Returns:
            float: Jaccard coefficient
        """
        support_union = rule['support']
        support_antecedent = rule['antecedent_support']
        support_consequent = rule['consequent_support']
        
        denominator = support_antecedent + support_consequent - support_union
        
        if denominator == 0:
            return 0.0
            
        jaccard = support_union / denominator
        return jaccard
        
    def filter_redundant_rules(self, similarity_threshold=0.8):
        """
        Remove redundant rules based on antecedent/consequent similarity
        
        Args:
            similarity_threshold (float): Threshold for considering rules similar
            
        Returns:
            list: Rules with redundant ones removed
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"FILTERING REDUNDANT RULES")
            print(f"{'='*60}")
            print(f"Similarity threshold: {similarity_threshold}")
            
        filtered = []
        
        for i, rule1 in enumerate(self.original_rules):
            is_redundant = False
            
            for j, rule2 in enumerate(filtered):
                similarity = self.calculate_rule_similarity(rule1, rule2)
                
                if similarity >= similarity_threshold:
                    # Keep the rule with higher confidence
                    if rule1['confidence'] <= rule2['confidence']:
                        is_redundant = True
                        break
                        
            if not is_redundant:
                filtered.append(rule1)
                
        if self.verbose:
            print(f"Rules before redundancy filtering: {len(self.original_rules)}")
            print(f"Rules after redundancy filtering: {len(filtered)}")
            print(f"Removed redundant rules: {len(self.original_rules) - len(filtered)}")
            
        return filtered
        
    def calculate_rule_similarity(self, rule1, rule2):
        """
        Calculate similarity between two rules based on items
        
        Args:
            rule1, rule2 (dict): Association rules to compare
            
        Returns:
            float: Similarity score (0.0 to 1.0)
        """
        # Get all items from both rules
        items1 = rule1['antecedent'].union(rule1['consequent'])
        items2 = rule2['antecedent'].union(rule2['consequent'])
        
        # Calculate Jaccard similarity
        intersection = len(items1.intersection(items2))
        union = len(items1.union(items2))
        
        if union == 0:
            return 0.0
            
        similarity = intersection / union
        return similarity
        
    def apply_multiple_filters(self, filters_config):
        """
        Apply multiple filters in sequence
        
        Args:
            filters_config (dict): Configuration for different filters
                Example: {
                    'lift': {'min': 1.1, 'max': 10.0},
                    'conviction': {'min': 1.2},
                    'leverage': {'min': 0.01},
                    'redundancy': {'threshold': 0.8}
                }
                
        Returns:
            list: Rules after applying all filters
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"APPLYING MULTIPLE FILTERS")
            print(f"{'='*80}")
            
        start_time = time.time()
        current_rules = self.original_rules.copy()
        filter_results = {}
        
        # Apply lift filter
        if 'lift' in filters_config:
            lift_config = filters_config['lift']
            self.original_rules = current_rules
            current_rules = self.filter_by_lift(
                min_lift=lift_config.get('min', 1.0),
                max_lift=lift_config.get('max', float('inf'))
            )
            filter_results['lift'] = len(current_rules)
            
        # Apply interest measures filter
        if 'conviction' in filters_config or 'leverage' in filters_config:
            self.original_rules = current_rules
            current_rules = self.filter_by_interest_measures(
                min_conviction=filters_config.get('conviction', {}).get('min', 1.0),
                min_leverage=filters_config.get('leverage', {}).get('min', 0.0)
            )
            filter_results['interest_measures'] = len(current_rules)
            
        # Apply redundancy filter
        if 'redundancy' in filters_config:
            self.original_rules = current_rules
            current_rules = self.filter_redundant_rules(
                similarity_threshold=filters_config['redundancy'].get('threshold', 0.8)
            )
            filter_results['redundancy'] = len(current_rules)
            
        self.filtered_rules = current_rules
        self.filter_stats = filter_results
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"MULTIPLE FILTERS RESULTS")
            print(f"{'='*80}")
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"Original rules: {len(self.original_rules)}")
            
            for filter_name, count in filter_results.items():
                print(f"After {filter_name}: {count} rules")
                
            print(f"Final filtered rules: {len(self.filtered_rules)}")
            reduction_percentage = ((len(self.original_rules) - len(self.filtered_rules)) / len(self.original_rules)) * 100
            print(f"Rules reduction: {reduction_percentage:.1f}%")
            
        return self.filtered_rules
        
    def categorize_rules_by_quality(self, rules=None):
        """
        Categorize rules into quality tiers
        
        Args:
            rules (list): Rules to categorize (uses filtered_rules if None)
            
        Returns:
            dict: Rules categorized by quality levels
        """
        if rules is None:
            rules = self.filtered_rules
            
        if not rules:
            print("No rules to categorize!")
            return {}
            
        categories = {
            'excellent': [],  # High confidence, lift, and support
            'good': [],       # Good metrics across the board
            'moderate': [],   # Decent rules but with limitations
            'weak': []        # Low quality rules
        }
        
        for rule in rules:
            confidence = rule['confidence']
            lift = rule['lift']
            support = rule['support']
            
            # Categorization logic
            if confidence >= 0.8 and lift >= 2.0 and support >= 0.05:
                categories['excellent'].append(rule)
            elif confidence >= 0.6 and lift >= 1.5 and support >= 0.03:
                categories['good'].append(rule)
            elif confidence >= 0.4 and lift >= 1.1 and support >= 0.01:
                categories['moderate'].append(rule)
            else:
                categories['weak'].append(rule)
                
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"RULE QUALITY CATEGORIZATION")
            print(f"{'='*60}")
            
            for category, rule_list in categories.items():
                print(f"{category.upper()}: {len(rule_list)} rules")
                
        return categories
        
    def display_filtered_rules(self, max_rules=15, sort_by='confidence'):
        """
        Display filtered rules in a formatted way
        
        Args:
            max_rules (int): Maximum number of rules to display
            sort_by (str): Sort criterion ('confidence', 'lift', 'support', 'conviction')
        """
        if not self.filtered_rules:
            print("No filtered rules to display!")
            return
            
        # Sort rules
        if sort_by == 'lift':
            sorted_rules = sorted(self.filtered_rules, key=lambda x: x['lift'], reverse=True)
        elif sort_by == 'support':
            sorted_rules = sorted(self.filtered_rules, key=lambda x: x['support'], reverse=True)
        elif sort_by == 'conviction':
            sorted_rules = sorted(self.filtered_rules, key=lambda x: x.get('conviction', 0), reverse=True)
        else:  # Default to confidence
            sorted_rules = sorted(self.filtered_rules, key=lambda x: x['confidence'], reverse=True)
            
        print(f"\n{'='*140}")
        print(f"TOP {min(max_rules, len(sorted_rules))} FILTERED RULES (sorted by {sort_by.upper()})")
        print(f"{'='*140}")
        
        # Header
        print(f"{'#':<3} {'RULE':<45} {'SUPP':<8} {'CONF':<8} {'LIFT':<8} {'CONV':<8} {'LEV':<8} {'JAC':<8}")
        print(f"{'-'*140}")
        
        # Display rules
        for i, rule in enumerate(sorted_rules[:max_rules]):
            antecedent_str = ', '.join(sorted(list(rule['antecedent'])))
            consequent_str = ', '.join(sorted(list(rule['consequent'])))
            
            # Truncate long item names
            if len(antecedent_str) > 18:
                antecedent_str = antecedent_str[:15] + "..."
            if len(consequent_str) > 18:
                consequent_str = consequent_str[:15] + "..."
                
            rule_str = f"{{{antecedent_str}}} -> {{{consequent_str}}}"
            
            conviction = rule.get('conviction', 0)
            leverage = rule.get('leverage', 0)
            jaccard = rule.get('jaccard', 0)
            
            print(f"{i+1:<3} {rule_str:<45} {rule['support']:<8.4f} {rule['confidence']:<8.4f} "
                  f"{rule['lift']:<8.4f} {conviction:<8.4f} {leverage:<8.4f} {jaccard:<8.4f}")

############################################
# Main execution function
############################################

def main():
    """
    Main function to run Step 4 of the Apriori algorithm
    """
    print("="*80)
    print("@APRIORI ALGORITHM PROJECT - ADVANCED RULE FILTERING")
    print("EDUARDO JUNQUEIRA - @EDAMI")
    print("="*80)
    print("STEP 4: ADVANCED RULE FILTERING AND QUALITY ASSESSMENT")
    print("="*80)
    
    # Initialize Rule Filter
    rule_filter = RuleFilter(verbose=True)
    
    # Load rules from Step 3
    success = rule_filter.load_rules_from_step3(limit=50, min_support=0.05, min_confidence=0.4)
    
    if not success:
        print("Error: Failed to load rules from Step 3. Please check previous steps.")
        return
    
    # Define filtering configuration
    filters_config = {
        'lift': {'min': 1.1, 'max': 10.0},       # Positive correlation only
        'conviction': {'min': 1.2},               # Strong conviction
        'leverage': {'min': 0.01},                # Significant leverage
        'redundancy': {'threshold': 0.7}          # Remove similar rules
    }
    
    # Apply multiple filters
    filtered_rules = rule_filter.apply_multiple_filters(filters_config)
    
    if filtered_rules:
        # Display filtered rules
        rule_filter.display_filtered_rules(max_rules=20, sort_by='lift')
        
        # Categorize rules by quality
        quality_categories = rule_filter.categorize_rules_by_quality()
        
        # Display excellent rules separately
        if quality_categories['excellent']:
            print(f"\n{'='*80}")
            print(f"EXCELLENT QUALITY RULES")
            print(f"{'='*80}")
            
            for i, rule in enumerate(quality_categories['excellent'][:10]):
                antecedent_str = ', '.join(sorted(list(rule['antecedent'])))
                consequent_str = ', '.join(sorted(list(rule['consequent'])))
                
                print(f"{i+1}. {{{antecedent_str}}} -> {{{consequent_str}}}")
                print(f"   Support: {rule['support']:.4f}, Confidence: {rule['confidence']:.4f}, "
                      f"Lift: {rule['lift']:.4f}, Conviction: {rule.get('conviction', 0):.4f}")
                print()
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY - STEP 4 COMPLETED")
    print(f"{'='*80}")
    print(f"Original rules from Step 3: {len(rule_filter.original_rules)}")
    print(f"Rules after advanced filtering: {len(filtered_rules)}")
    
    if rule_filter.filter_stats:
        for filter_name, count in rule_filter.filter_stats.items():
            print(f"After {filter_name} filter: {count} rules")
    
    print(f"\nStep 4 completed successfully!")
    print(f"High-quality association rules identified and filtered!")

############################################
# Execute main function if script is run directly
############################################

if __name__ == "__main__":
    main()