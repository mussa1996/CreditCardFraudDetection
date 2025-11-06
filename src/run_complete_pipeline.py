# run_complete_pipeline.py
import os
import sys
from src.train import train
from src.evaluate_comprehensive import evaluate_comprehensive

def run_complete_pipeline():
    """Run complete training and evaluation pipeline"""
    
    print("🚀 Starting Complete Pipeline...")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Train the model
    print("\n" + "="*50)
    print("STEP 1: TRAINING HYBRID LSTM MODEL")
    print("="*50)
    history, model = train()
    
    # Step 2: Comprehensive evaluation
    print("\n" + "="*50)
    print("STEP 2: COMPREHENSIVE EVALUATION")
    print("="*50)
    results = evaluate_comprehensive()
    
    # Step 3: Summary
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Final Model Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"AUC Score: {results['auc_score']:.4f}")
    print("\nNext steps:")
    print("1. Run 'python simulate_transactions.py' to test real-time predictions")
    print("2. Start API with: 'uvicorn app:app --reload --port 8000'")
    print("3. Open dashboard with: 'streamlit run dashboard.py'")

if __name__ == "__main__":
    run_complete_pipeline()