FedMed — Implementation Progress= [18-08-26]
1. Project Environment Setup
Created the FED_MED project directory.

Created and activated a Python virtual environment:

.venv
Current environment initially used Python 3.14.5 (64-bit).
2. Federated Learning Dependencies

Installed the required Federated Learning framework:

Flower (flwr) 1.33.0
FastAPI and supporting Flower dependencies
gRPC
Protobuf
Cryptography
PyYAML
SQLAlchemy and other Flower dependencies

Flower installation completed successfully.

3. PyTorch Setup

Attempted to install:

PyTorch
TorchVision
NumPy
Pillow and supporting libraries

The initial installation used:

torch 2.13.0
torchvision 0.28.0

but importing PyTorch produced:

OSError: [WinError 1114]
Dynamic link library (DLL) initialization routine failed

specifically while loading:

torch\lib\c10.dll

4. CPU PyTorch Attempt

The regular PyTorch installation was removed and a CPU-only build was installed:

torch 2.13.0+cpu
torchvision 0.28.0+cpu

The installation itself completed successfully.

However, the same c10.dll initialization issue remained during import.

5. Current Status
Environment
    ↓
Python 3.14.5
    ↓
Virtual Environment (.venv)
    ↓
Flower 1.33.0                 ✅
    ↓
PyTorch                       ⚠️ DLL initialization issue
    ↓
ML Model                      ⏳
    ↓
Federated Clients             ⏳
    ↓
FedAvg Server                 ⏳
6. Next Step

The next action is to create a Python 3.12 virtual environment and install PyTorch + Flower there.

After PyTorch imports successfully, implementation will proceed in this order:

PyTorch Model
      ↓
Hospital Client 1
Hospital Client 2
Hospital Client 3
      ↓
Flower Server
      ↓
FedAvg Aggregation
      ↓
Global Model

Important: No actual FedMed ML/FL model has been implemented yet; so far, we have completed environment setup and dependency installation/troubleshooting.