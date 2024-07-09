# Qserv_dataset

The Qserv dataset collects the embeddings of 35000 random quantum circuits (executed in Hanoi and Cairo backends) in two separate ways with the purpose of approaching the problem of quantum error estimation by employing machine learning and neural network methods.

## Gate counts embedding

The first embedding method provides a definition of each circuit by a quantum gate count. Besides the gate count, the dataset also contains the following columns:

➥ Error_Hanoi: Calculated error with the Hanoi backend  
➥ Error_Cairo: Calculated error with the Cairo backend  
➥ circuit: Index of the circuit  
➥ num_qubits: Number of qubits that the circuit contains  
➥ depth: Depth of the circuit

The studies conducted with this approach do not make use of the prevously mentioned columns in the training process.

```python
import pandas as pd

df = pd.read_csv("circuits_complete.csv")

columns = df.columns.tolist()
columns.remove("Error_Hanoi")
columns.remove("Error_Cairo")
columns.remove("circuit")
columns.remove("num_qubits")
columns.remove("depth")

X = df[columns]
y = df["Error_Hanoi"] # or df["Error_Cairo"]
```

## Tabular data adaptation

The second embedding method provides a definition of each circuit by representing them with a tabular-based projection.

```python
backend = "hanoi" # or "cairo"
num_qubit = 4 # or any other number form 4 to 10 

data = torch.load(f"data({num_qubit}q-{backend}).pt")
X, y = data.tensors
```
