import torch
from rdkit import Chem
from torch_geometric.data import Data
from your_model_file import ImprovedGCN  # Import your model class

# 1. Load model architecture and weights
def load_model(model_path='best_model.pt'):
    # Initialize model (replace num_features with your actual value)
    model = ImprovedGCN(num_features=your_num_features)  
    
    # Load saved weights
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 2. SMILES to Data conversion (customize with your features)
def smiles_to_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Atom features example - REPLACE with your actual features
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            float(atom.GetAtomicNum()),
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetIsAromatic())
        ]
        atom_features.append(features)
    
    # Edge connections
    edge_index = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])  # Undirected graph
    
    if not edge_index:  # Single atom case
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(
        x=torch.tensor(atom_features, dtype=torch.float),
        edge_index=edge_index
    )

# 3. Prediction function
def predict(model, smiles):
    data = smiles_to_data(smiles)
    if data is None:
        return "Invalid SMILES string"
    
    # Create batch (single molecule)
    batch = torch.zeros(data.num_nodes, dtype=torch.long)
    
    with torch.no_grad():
        prediction, _ = model(data.x, data.edge_index, batch)
    
    return f"Predicted logD: {prediction.item():.2f}"

# 4. Interactive prediction loop
def main():
    try:
        model = load_model()
        print("Molecular Property Predictor (type 'quit' to exit)")
        
        while True:
            smiles = input("\nEnter SMILES string: ").strip()
            if smiles.lower() in ['quit', 'exit', 'q']:
                break
            
            print(predict(model, smiles))
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Make sure:")
        print("- You have 'best_model.pt' in this directory")
        print("- Your model class is properly imported")
        print("- Feature dimensions match training")

if __name__ == "__main__":
    main()