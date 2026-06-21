import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from validator import DataValidator

# Create mock data instantly so the user doesn't need to download anything
print("Generating dummy baseline data (real_data.csv alternative)...")
mock_dict = {
    'Age': np.random.randint(18, 70, size=500),
    'Income': np.random.normal(55000, 15000, size=500),
    'Score': np.random.uniform(0, 100, size=500)
}
real_data = pd.DataFrame(mock_dict)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)

print("Training the Neural Network (CTGAN)... This will take a few seconds...")
synthesizer = CTGANSynthesizer(metadata, epochs=50)
synthesizer.fit(real_data)

print("Generating synthetic data...")
synthetic_data = synthesizer.sample(num_rows=500)

validator = DataValidator(threshold=0.05)
numerical_cols = ['Age', 'Income', 'Score']
report = validator.evaluate(real_data, synthetic_data, numerical_cols)

print("\n=== VALIDATION REPORT FOR PROFESSOR ===")
for col, metrics in report.items():
    status = "❌ FAIL (Drift Detected)" if metrics['retrain'] else "✅ PASS (High Fidelity)"
    print(f"Column: {col.ljust(8)} | p-value: {metrics['p_value']:.4f} | JS Div: {metrics['js_divergence']:.4f} | Status: {status}")

synthetic_data.to_csv('synthetic_output.csv', index=False)
print("\nSuccess! Results saved to synthetic_output.csv")