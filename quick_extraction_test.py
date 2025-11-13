"""
Quick test script for model extraction attack
Tests with minimal data to verify the setup works correctly
"""

import subprocess
import sys

def run_quick_test():
    """Run a quick test with minimal data"""
    
    print("="*80)
    print("QUICK EXTRACTION ATTACK TEST")
    print("="*80)
    print()
    print("This will run a quick test with:")
    print("- 100 training samples")
    print("- 50 test samples")
    print("- 1 epoch")
    print("- Small batch size")
    print()
    print("This should take only a few minutes.")
    print()
    
    cmd = [
        sys.executable, "extraction_attack.py",
        "--victim_model", "gpt2",
        "--adversary_model", "distilgpt2",
        "--dataset", "wikitext-2-raw-v1",
        "--train_samples", "100",
        "--test_samples", "50",
        "--batch_size", "8",
        "--epochs", "1",
        "--learning_rate", "5e-5",
        "--temperature", "1.0",
        "--alpha", "0.5",
        "--output_dir", "extraction_results_test",
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print()
        print("="*80)
        print("QUICK TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        print()
        print("Results saved in: extraction_results_test/")
        print()
        print("To run full experiments, use:")
        print("  Windows: run_extraction.bat")
        print("  Linux/Mac: ./run_extraction.sh")
        print()
        
    except subprocess.CalledProcessError as e:
        print()
        print("="*80)
        print("TEST FAILED!")
        print("="*80)
        print(f"Error: {e}")
        print()
        print("Please check:")
        print("1. All required packages are installed")
        print("2. Python environment is set up correctly")
        print("3. Internet connection for downloading models/data")
        print()
        sys.exit(1)

if __name__ == "__main__":
    run_quick_test()

