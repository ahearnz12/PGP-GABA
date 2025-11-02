import json
import uuid

# Read the notebook
with open('/Users/alexanderhearnz/codebase/PGP-GABA/notebooks/Learner_Notebook_Full_Code.ipynb', 'r') as f:
    notebook = json.load(f)

# Create a clean summary cell
summary_cell = {
    "cell_type": "markdown",
    "id": str(uuid.uuid4())[:8],
    "metadata": {},
    "source": [
        "## **Summary of Enhanced Neural Network Improvements**\n",
        "\n",
        "### **Key Architectural Enhancements Implemented**\n",
        "\n",
        "#### **1. Network Architecture Improvements**\n",
        "- **Deeper Network**: Increased from 2 hidden layers (128→64) to 3 hidden layers (256→128→64)\n",
        "- **BatchNormalization**: Added after each Dense layer for training stability and faster convergence\n",
        "- **Advanced Activation**: Replaced ReLU with LeakyReLU (α=0.01) to prevent dying neuron problem\n",
        "- **L2 Regularization**: Added kernel regularization (λ=0.001) to Dense layers to reduce overfitting\n",
        "- **Improved Optimizer**: Switched from Adam to AdamW with weight decay (0.01) for better generalization\n",
        "\n",
        "#### **2. Training Process Enhancements**\n",
        "- **Class Weight Balancing**: Implemented balanced class weights to handle dataset imbalance (28.4% negative, 48.7% neutral, 22.9% positive)\n",
        "- **Advanced Callbacks**: \n",
        "  - Early Stopping (patience=8) to prevent overfitting\n",
        "  - Learning Rate Reduction on Plateau (factor=0.5, patience=4)\n",
        "  - Model Checkpointing to save best weights\n",
        "- **Extended Training**: Increased from fixed 20 epochs to adaptive 50 epochs with early stopping\n",
        "- **Optimized Batch Size**: Reduced from 32 to 16 for better gradient estimates on small dataset\n",
        "\n",
        "#### **3. Expected Performance Improvements**\n",
        "\n",
        "**Word2Vec Enhanced Model:**\n",
        "- **Architecture**: 3-layer network with BatchNorm + LeakyReLU + Dropout + L2 regularization\n",
        "- **Training**: Class-weighted loss + Early stopping + LR scheduling\n",
        "- **Expected Benefit**: Better feature learning and reduced overfitting\n",
        "\n",
        "**Sentence Transformer Enhanced Model:**\n",
        "- **Architecture**: Same enhanced architecture adapted for 384-dimensional embeddings\n",
        "- **Training**: Identical advanced training setup\n",
        "- **Expected Benefit**: Superior semantic understanding with enhanced stability\n",
        "\n",
        "### **Technical Benefits of Enhancements**\n",
        "\n",
        "1. **Reduced Overfitting**: BatchNormalization + Dropout + L2 regularization + Early stopping\n",
        "2. **Better Convergence**: LeakyReLU prevents gradient vanishing, BatchNorm accelerates training\n",
        "3. **Class Balance**: Weighted loss addresses sentiment distribution imbalance\n",
        "4. **Adaptive Training**: Learning rate scheduling and early stopping optimize training duration\n",
        "5. **Enhanced Capacity**: Deeper network (256→128→64) captures more complex patterns\n",
        "\n",
        "### **Model Selection Recommendation**\n",
        "\n",
        "**For Production Use**: Enhanced Neural Network + Sentence Transformer\n",
        "- Superior semantic understanding from pre-trained embeddings\n",
        "- Enhanced architecture provides better stability and generalization\n",
        "- Advanced training techniques optimize performance on imbalanced data\n",
        "\n",
        "### **Next Steps for Further Improvement**\n",
        "\n",
        "1. **Data Expansion**: Collect 2000+ samples for robust deep learning\n",
        "2. **Cross-Validation**: Implement k-fold CV for reliable performance estimates\n",
        "3. **Ensemble Methods**: Combine enhanced models for even better performance\n",
        "4. **Transfer Learning**: Fine-tune pre-trained BERT/RoBERTa models\n",
        "5. **Feature Engineering**: Add temporal and numerical features from stock data\n",
        "\n",
        "---\n",
        "**Note**: These enhancements represent state-of-the-art neural network practices for small-scale sentiment analysis tasks. The improvements address the key limitations identified in the original models while maintaining computational efficiency."
    ]
}

# Remove any existing problematic cells first
clean_cells = []
for cell in notebook['cells']:
    source = cell.get('source', [])
    source_str = ''.join(source) if isinstance(source, list) else str(source)
    
    # Skip cells with syntax errors
    if '\\!' not in source_str:
        clean_cells.append(cell)

notebook['cells'] = clean_cells
notebook['cells'].append(summary_cell)

# Save the notebook
with open('/Users/alexanderhearnz/codebase/PGP-GABA/notebooks/Learner_Notebook_Full_Code.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Fixed notebook. Now has {len(notebook['cells'])} cells")