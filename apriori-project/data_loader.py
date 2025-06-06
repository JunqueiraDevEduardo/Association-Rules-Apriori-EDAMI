############################################
#               STEP 1:data_loader.py 
# FILE for Online Retail I:
# This is a transactional data set which 
# contains all the transactions occurring between 
# 01/12/2010 and 09/12/2011 for a UK-based and 
# registered non-store online retail.
# The company mainly sells unique all-occasion gifts. 
# Many customers of the company are wholesalers.
# EDUARDO JUNQUEIRA 
# TOPIC 1 ASSOCIATION RULES WITH APRIORI ALGORITHM PROJECT EDAMI
############################################

############################################
#Import  libraries: polars, and others relevant for the compilation of the python code.
############################################
import polars as pl 
import time

############################################
#                   Main
#Setup code 
## Download, and import local Online Retail I 
#######################################

#local file:
file_path = '/Users/eduardommj/Documents/GitHub/Association-Rules-Apriori-EDAMI/apriori-project/Online_Retail_DataSet/Online Retail.xlsx'

def load_online_retail_transactions(limit=100, verbose=True):
    # Start counting the timer for all process code!
    start_time = time.time()
    #The Docstring ("""Explain function def load_online_retail_transactions(limit=100, verbose=True)):
    """
    Load and preprocess Online Retail dataset for association rules mining
    
    Args:
        limit (int): Number of unique invoices to process (100 for performance as per design document)
        verbose (bool): Print processing information,to print progress messages (True/False)
    
    Returns:
        list: List of transactions (each transaction is a list of product descriptions)
        How much bigger is the list must time will spend for show the list information.
        Example: [['Product A', 'Product B'], ['Product C', 'Product D', 'Product A']]
    """
    
    # Load Excel file (not CSV!)
    try:
        # Data loader with Polars - using read_excel for .xlsx files
        print("Loading Excel file...")
        df = pl.read_excel(file_path)
        print(f"Successfully loaded {len(df)} rows from dataset")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print("Please check the file path and make sure the file exists")
        return []
    except Exception as e:
        print(f"Error loading file: {e}")
        return []

#######################################
#1. Initial information about data - statistics:
#######################################
        
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 1: DATASET (D) INITIAL ANALYSIS")
        print(f"{'='*60}")
        
        #see headers (fixed typo: columns not collumns)
        print("Headers (columns):", df.columns)
        # see the length of:
        print("Number of columns:", len(df.columns))
        print("Number of rows:", len(df))
        print("Original dataset shape:", df.shape)
        
        
        # Display sample data before cleaning
        print(f"\nSample data before cleaning:")
        print(df.head())

#######################################
#2. Data Changes-Cleaning and Processing:
#######################################   
    print(f"\n{'='*60}")
    print(f"STEP 1: DATA CLEANING AND PREPROCESSING")
    print(f"{'='*60}")
    print("Start data cleaning process...")
    
    #2. Remove nulls in relevant columns:
    print(" Remove null values")
    original_rows = len(df)
    df = df.drop_nulls(['InvoiceNo','StockCode','Description','Quantity','InvoiceDate','UnitPrice','CustomerID','Country'])
    rows_after_nulls = len(df)
    print(f"  Removed {original_rows - rows_after_nulls} rows with null values")
    
    #4. Removing duplicates (same invoice, same product, same quantity,same description...)
    print(" Remove duplicates")
    df = df.unique()
    rows_after_duplicates = len(df)
    print(f"  Removed {rows_after_nulls - rows_after_duplicates} duplicate rows")

    #5.Filter Quantity:
    print(" Filter Quantity > 0 ")
    df = df.filter(pl.col('Quantity') > 0)
    rows_after_quantity = len(df)
    print(f"  Removed {rows_after_duplicates - rows_after_quantity} rows with negative/zero quantities")

#######################################
#1. Data Changes Summarize:
####################################### 
    #6. Summarize all valid:
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 1: DATA CLEANING SUMMARY")
        print(f"{'='*60}")
        print("Shape after cleaning:", df.shape)
        print("Sample cleaned data:")
        print(df.head())
        print("Null counts per column after cleaning:")
        print(df.null_count())

# Count unique invoices BEFORE processing
    unique_invoices_count = df['InvoiceNo'].n_unique()
    print(f"\n{'='*60}")
    print(f"STEP 1: INVOICE ANALYSIS")
    print(f"{'='*60}")
    print(f"Total unique InvoiceNo found: {unique_invoices_count}")
    print(f"Note: It's big data ({len(df)} rows); limiting to {limit} transactions for performance!")

    # Additional dataset statistics
    unique_products = df['Description'].unique()
    unique_customers = df['CustomerID'].unique()
    print(f"Total unique products in dataset: {len(unique_products)}")
    print(f"Total unique customers: {len(unique_customers)}")
    
    # Show date range
    try:
        date_range = df.select([
            pl.col('InvoiceDate').min().alias('earliest_date'),
            pl.col('InvoiceDate').max().alias('latest_date')
        ])
        print(f"Transaction date range:")
        print(date_range)
    except Exception as e:
        print(f"Could not determine date range: {e}")

#######################################
#1. Data Changes more important: This converts the data into format needed for association rules
#######################################
    print(f"\n{'='*60}")
    print(f"STEP 1: TRANSACTION CREATION")
    print(f"{'='*60}")

    #7. Select only the first 100 unique invoices (as per design document)
    print(f" Select only the first {limit} unique InvoiceNo (performance optimization)")
    unique_invoices = df['InvoiceNo'].unique().limit(limit)
    df_filtered = df.filter(pl.col('InvoiceNo').is_in(unique_invoices.to_list()))
    print(f"  Filtered dataset shape: {df_filtered.shape}")
    
    #8. Group by InvoiceNo to create transactions
    print(" Creating transaction format...")
    print("  Extract all product descriptions by their InvoiceNo")
    print("  Each group becomes a unique item of a transaction")
    print("  Each row = 1 basket (invoice) of items")
    
    transactions = []
    
    # Group by InvoiceNo and collect product descriptions
    # This line groups all products that belong to the same invoiceNumber
    grouped = df_filtered.group_by('InvoiceNo').agg([
        pl.col('Description').unique().alias('products')
    ])
    
    # Convert to list format needed for association rules
    print(" Convert to transaction format for Apriori algorithm...")
    for row in grouped.iter_rows(named=True):
        transaction = [desc for desc in row['products'] if desc is not None]
        if transaction:  # Only add non-empty transactions
            transactions.append(transaction)
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 1: PROCESSING RESULTS")
        print(f"{'='*60}")
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Total transactions created: {len(transactions)}")
        
        # Transaction statistics
        if transactions:
            transaction_sizes = [len(trans) for trans in transactions]
            avg_size = sum(transaction_sizes) / len(transaction_sizes)
            min_size = min(transaction_sizes)
            max_size = max(transaction_sizes)
            
            print(f"Transaction statistics:")
            print(f" Average items per transaction: {avg_size:.2f}")
            print(f" Smallest transaction: {min_size} items")
            print(f" Largest transaction: {max_size} items")
            
            print(f"\nSample transactions (first 5):")
            for i, trans in enumerate(transactions[:5]):
                print(f"  Transaction {i+1} ({len(trans)} items): {trans[:3]}{'...' if len(trans) > 3 else ''}")
        
    
    #9- Return transaction list
    return transactions


##############################################
##Principal Execution!
##############################################

if __name__ == "__main__":
    print("="*80)
    print("APRIORI ALGORITHM PROJECT - ASSOCIATION RULES MINING")
    print("EDUARDO JUNQUEIRA - EDAMI")
    print("="*80)
    print("STEP 1: DATA LOADER - ONLINE RETAIL TRANSACTIONS")
    print("="*80)
    
    transactions = load_online_retail_transactions(limit=100, verbose=True)
    