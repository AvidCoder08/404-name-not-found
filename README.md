# Money Muling Detection Engine 🕵️‍♂️💸

## Graph-Based Financial Crime Detection

This solution uses **Graph Neural Networks (GNNs)** and **Advanced Graph Algorithms** to detect sophisticated money laundering patterns, specifically **Money Muling Rings**.

### Key Features
-   **Synthetic Data Generator**: Creates realistic financial transaction logs with injected fraud patterns (Cycles, Smurfing).
-   **GNN Suspicion Scoring**: Uses GraphSAGE to learn suspicious account embeddings.
-   **Pattern Detection**: Deterministic detection of:
    -   Circular Flows (Mule Rings)
    -   Smurfing (Fan-in/Fan-out)
    -   Layered Shells
-   **Interactive Dashboard**: Streamlit-based UI for analysis and visualization.

### Installation

```bash
pip install -r requirements.txt
```

### Usage

1.  **Generate Data**:
    ```bash
    python3 scripts/generate_data.py
    ```

2.  **Run Application**:
    ```bash
    streamlit run app.py
    ```

### Tech Stack
-   **Python 3.10+**
-   **PyTorch & PyTorch Geometric**: GNN Model
-   **NetworkX**: Graph Analysis
-   **Streamlit**: Web Interface
-   **Pandas & Scikit-Learn**: Data Handling
