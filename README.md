

# FedMed

## Privacy-Preserving Cross-Silo Federated Learning Engine for Medical AI

FedMed is a modular federated learning engine designed for privacy-preserving collaborative machine learning across distributed medical institutions.

The system separates the framework-independent federated learning core from the Flower runtime layer, allowing the core training, strategy, aggregation, data, and model components to remain independent of the federated-learning framework.

---

# 1. Project Overview

FedMed is designed around a cross-silo federated learning architecture where multiple medical organizations can collaboratively train a machine learning model without directly sharing their local training data.

The current implementation focuses on:

- Federated client-server training
- Federated rounds
- Model parameter exchange
- FedAvg aggregation
- Client selection
- Client failure handling
- IID and label-skew data partitioning
- Local training configuration
- Data quantity imbalance
- Centralized vs federated experimentation
- Flower 1.34.0 runtime integration
- Deterministic and reproducible experiments
- Automated testing

The project is currently being developed as a modular research and experimentation platform.

---

# 2. Architecture

FedMed follows a framework-independent core architecture with Flower isolated at the application boundary.

```text
                         FedMed
                           |
              +------------+------------+
              |                         |
       Framework-Independent       Flower Runtime
             Core                    Boundary
              |                         |
              v                         v
        Federated Client          app/client.py
        Federated Strategy        app/server.py
        Aggregator                app/main.py
        Round Coordinator
        Parameters
        Training
        Evaluation
        Data
        Models
              |
              v
        FedAvg Aggregation

The high-level structure is:

FedMed
├── src/
│   ├── common/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── aggregation/
│   ├── fl/
│   └── monitoring/
│
├── app/
│   ├── client.py
│   ├── server.py
│   ├── failure_mod.py
│   └── main.py
│
├── configs/
├── tests/
├── pyproject.toml
├── requirements.txt
├── Dockerfile
└── docker-compose.yml


---

3. Core Design Principles

3.1 Framework Independence

The main federated learning logic remains under:

src/

The core does not depend on Flower transport or Flower message structures.

Flower-specific behavior remains under:

app/

This keeps the federated learning engine separated from the runtime framework.


---

3.2 Strategy and Aggregator Separation

FedMed intentionally separates federation policy from mathematical aggregation.

FedMed Strategy
       |
       | federation policy
       | client selection
       | round decisions
       v
Aggregator
       |
       | mathematical aggregation
       v
Aggregated Parameters

The responsibility split is:

Strategy
    ├── client selection
    ├── federation policy
    ├── round-level decisions
    └── delegates aggregation

Aggregator
    └── mathematical parameter aggregation

The current implementation uses:

FedAvgStrategy
        |
        v
FedAvgAggregator

The Flower adapter does not duplicate FedAvg mathematics.


---

4. Flower Runtime Architecture

FedMed currently integrates with:

Flower 1.34.0

The Flower runtime boundary is:

Flower ServerApp
       |
       v
FedMedFlowerStrategy
       |
       v
FedMed FedAvgStrategy
       |
       v
FedAvgAggregator

The Flower adapter is implemented in:

app/server.py

The client adapter is implemented in:

app/client.py

The application composition root is:

app/main.py

The framework-independent composition root is:

src/fl/orchestrator.py


---

5. Flower Server Adapter

app/server.py contains the Flower-side server adapter.

The main class is:

FedMedFlowerStrategy

Its responsibilities include:

Flower node selection

Training message construction

Evaluation message construction

Flower Message handling

Training-result conversion

Evaluation-result conversion

Delegation to the FedMed strategy

Handling failed client replies

Aggregating successful client results

Returning Flower-compatible responses


The Flower adapter does not implement FedAvg mathematics directly.


---

6. Training Flow

The federated training flow is:

Flower ServerApp
       |
       v
configure_train()
       |
       +--> Select Flower nodes
       |
       +--> Read global parameters
       |
       +--> Create training configuration
       |
       +--> Add server round
       |
       +--> Construct Flower Messages
       |
       v
Flower SuperNode
       |
       v
Flower ClientApp
       |
       v
FedMed FederatedClient
       |
       v
Local Training
       |
       v
Training Result
       |
       v
Flower Server
       |
       v
aggregate_train()
       |
       v
FedMed Strategy
       |
       v
FedAvgAggregator
       |
       v
New Global Parameters


---

7. Evaluation Flow

The evaluation flow is:

Flower ServerApp
       |
       v
configure_evaluate()
       |
       +--> Select evaluation nodes
       |
       +--> Create evaluation configuration
       |
       +--> Construct EVALUATE messages
       |
       v
Flower ClientApp
       |
       v
Local Evaluation
       |
       v
Evaluation Result
       |
       v
Flower Server
       |
       v
aggregate_evaluate()
       |
       v
FedMed Strategy
       |
       v
Aggregated Evaluation Metrics


---

8. Flower Compatibility

FedMed was verified against:

Flower 1.34.0

The Flower runtime uses:

arrays
config

as the standard record keys.

Training messages therefore contain:

arrays
config

Evaluation messages use the same record structure.

The Flower adapter converts Flower results into FedMed result objects.

For training:

Flower FitRes
      |
      v
FederatedFitResult

For evaluation:

Flower EvaluateRes
      |
      v
FederatedEvaluateResult

The conversion layer validates:

parameter payloads

number of examples

metric types

numeric metric values

result status


Parameter arrays are copied at the Flower/FedMed boundary to avoid accidental mutation between the two layers.


---

9. FedMed Orchestrator

The composition root is:

src/fl/orchestrator.py

FedMedOrchestrator assembles:

models

training configuration

evaluation

data partitions

federated clients

strategies

aggregators

Flower ClientApp

Flower ServerApp


The orchestrator provides:

build_client()
build_strategy()
build_client_app()
build_server_app()
build_apps()

The orchestrator now uses the FedMed data-loader abstraction:

src.data.loader.create_dataloader

instead of constructing a raw PyTorch DataLoader directly.


---

10. Data Partitioning

FedMed supports:

IID
Label Skew

The partitioning logic is implemented in:

src/data/partitioner.py

The partitioner provides deterministic client partitions using configured seeds.

Example:

Global Dataset
      |
      v
Partitioner
      |
      +---- Client 0
      |
      +---- Client 1
      |
      +---- Client 2
      |
      +---- Client 3

The project includes tests covering the partitioning behavior.


---

11. Client Training

Each federated client contains:

Model
Trainer
Evaluator
Training Data
Evaluation Data
Client ID

The client performs:

Receive Global Parameters
          |
          v
Load Parameters
          |
          v
Local Training
          |
          v
Return Updated Parameters
          |
          v
Local Evaluation

The current training implementation uses PyTorch.

The smoke-test model is:

FlowerSmokeTestModel


---

12. Client Failure Handling

FedMed includes controlled client-failure experimentation.

The Flower client runtime supports an E6 controlled dropout mechanism through:

app/failure_mod.py

The purpose is to test whether a failed training client prevents the remaining clients from completing a federated round.

The server handles failed training replies by ignoring the failed client and continuing aggregation using successful clients.


---

13. Experimental Methodology

Each experiment follows a controlled workflow:

1. Define objective
       |
2. Fix unrelated configuration
       |
3. Change one experimental variable
       |
4. Run federated workload
       |
5. Record metrics
       |
6. Record parameter fingerprints
       |
7. Record runtime evidence
       |
8. Document observations

Experiments are treated as observations of the current configuration rather than universal conclusions.


---

14. Federated Learning Experiments

The current completed experiments are:

E1 — Baseline Reproducibility
E2 — Client Count
E3 — Training Client Participation Fraction
E4 — Number of Federated Rounds
E5 — IID vs Non-IID Data
E6 — Client Failure / Dropout
E7 — Local Epochs
E8 — Data Quantity Imbalance
E9 — Centralized vs Federated Training


---

15. E1 — Baseline Reproducibility

Objective

Verify that the same FedMed configuration produces reproducible federated-learning results across repeated executions.

Configuration:

Clients: 2
Rounds: 3
Train fraction: 1.0
Evaluation fraction: 1.0
Local epochs: 1
Partition: IID
Strategy: FedAvgStrategy
Aggregator: FedAvgAggregator

The experiment was executed twice using the same configuration.

Parameter fingerprints:

Round 1:
5d2399307f878547
        ->
166dbbaac8c674b6

Round 2:
166dbbaac8c674b6
        ->
2c9f2041b7a13ade

Round 3:
2c9f2041b7a13ade
        ->
32346c94ed9fdb6f

Results:

Train loss:
R1 0.8096474260
R2 0.8063939661
R3 0.8031985164

Evaluation loss:
R1 0.6489310861
R2 0.6493559479
R3 0.6498010904

Accuracy:
50.00% every round

Examples:
16 per round

The repeated runs produced matching parameter fingerprints and metrics, validating deterministic behavior for this baseline configuration.


---

16. E2 — Client Count

Objective

Observe federated-learning behavior as the number of participating Flower clients changes.

All experiments used:

Rounds: 3
Training fraction: 100%
Evaluation fraction: 100%

E2-A — 1 Client

Training clients: 1
Evaluation clients: 1
Examples per round: 8

Train loss:
0.8189847767
0.8152351081
0.8115414977

Evaluation loss:
0.6616895199
0.6618472934
0.6620339751

Accuracy:
50.00%

E2-B — 2 Clients

Training clients: 2
Evaluation clients: 2
Examples per round: 16

Train loss:
0.8096474260
0.8063939661
0.8031985164

Evaluation loss:
0.6489310861
0.6493559479
0.6498010904

Accuracy:
50.00%

E2-C — 3 Clients

Training clients: 3
Evaluation clients: 3
Examples per round: 24

Train loss:
0.7625652552
0.7611813347
0.7598178188

Evaluation loss:
0.6903412938
0.6900853515
0.6898385584

Accuracy:
41.67%

The experiment changes both client count and total participating data because each client contributes 8 examples. Therefore, the experiment does not isolate client count as a completely independent variable.


---

17. E3 — Training Client Participation Fraction

Objective

Validate client participation fraction handling in the real Flower multi-node runtime.

Configuration:

Available clients: 4
Rounds: 3
Evaluation fraction: 1.0
Local epochs: 1

Results

Training Fraction	Training Clients	Evaluation Clients

100%	4	4
75%	3	4
50%	2	4
25%	1	4


The Flower runtime correctly selected:

100% -> 4 clients
75%  -> 3 clients
50%  -> 2 clients
25%  -> 1 client

Observed accuracies:

100%:
43.75%, 43.75%, 43.75%

75%:
43.75%, 43.75%, 43.75%

50%:
56.25%, 56.25%, 59.375%

25%:
56.25%, 59.375%, 59.375%

These results describe the particular deterministic client-selection configuration and should not be interpreted as evidence that lower participation is inherently better.


---

18. E4 — Number of Federated Rounds

Objective

Examine the behavior of the system across an increased number of federated rounds.

Configuration:

Clients: 4
Rounds: 5
Training participation: 100%
Evaluation participation: 100%

Results:

Round 1:
Train loss = 0.6799688861
Eval loss  = 0.7347568944
Accuracy   = 0.46875

Round 2:
Train loss = 0.6799019128
Eval loss  = 0.7345721349

Round 3:
Train loss = 0.6798361465
Eval loss  = 0.7343898416

Round 4:
Train loss = 0.6797715276
Eval loss  = 0.7342100814

Round 5:
Train loss = 0.6797080487
Eval loss  = 0.7340327278

Accuracy remained:

46.875%

for all five rounds.

The experiment confirms successful execution across five federated rounds.


---

19. E5 — IID vs Label-Skew Data

Objective

Compare the federated workflow under IID and label-skew data partitioning.

Configuration:

Clients: 4
Rounds: 3
Training participation: 100%
Evaluation participation: 100%

E5-A — IID

Train loss:
0.6870625988
0.6868757978
0.6866935045

Eval loss:
0.6867602393
0.6865770072
0.6863982305

Accuracy:
50.00%

E5-B — Label Skew

Train loss:
0.6842055842
0.6840200424
0.6838389635

Eval loss:
0.6867608801
0.6865782812
0.6864000931

Accuracy:
50.00%

Both partitioning configurations completed successfully in the real Flower runtime.

The experiment confirms that the FedMed partitioning and aggregation pipeline can execute with both IID and label-skew client distributions.


---

20. E6 — Client Failure / Dropout

Objective

Test federated-round resilience when one training client fails.

Configuration:

Flower SuperNodes: 4
Partitions: 0, 1, 2, 3
Training participation: 100%
Evaluation participation: 100%
Rounds: 3
Data partitioning: IID
Controlled failure: Partition 0 during Round 2
Failure scope: TRAIN only

Results:

Round	Successful Training Clients	Training Examples	Train Loss	Eval Loss	Accuracy

1	4/4	32	0.687063	0.686760	0.5000
2	3/4	24	0.701005	0.686616	0.5000
3	4/4	32	0.686734	0.686436	0.5000


During Round 2, the controlled client failure was triggered:

E6 controlled client dropout: partition=0, round=2

The server ignored the failed training reply and continued aggregation using the successful clients.

Round 2 therefore completed with:

3 successful training clients
24 training examples

Round 3 returned to:

4 successful training clients
32 training examples

This demonstrates the implemented client-failure tolerance path in the Flower runtime.


---

21. E7 — Local Epochs

Objective

Study the effect of different local training epoch configurations while keeping the federated runtime configuration fixed.

Configurations:

E7-A -> 1 local epoch
E7-B -> 2 local epochs
E7-C -> 5 local epochs

Fixed configuration:

Flower SuperNodes: 4
Training participation: 100%
Evaluation participation: 100%
Rounds: 3
Partitioning: IID
Batch size: 4
Learning rate: 0.01
Optimizer: SGD
Seed: 42
Controlled failure: disabled

E7 Summary

Metric	1 Epoch	2 Epochs	5 Epochs

Final train loss	0.686694	0.682896	0.672117
Final eval loss	0.686398	0.685882	0.684518
Accuracy	0.5000	0.5000	0.5000
Train examples/round	32	64	160
Batches/client	2	4	10
Runtime	250.32s	250.24s	311.00s


All three configurations completed all three federated rounds.

The experiments showed lower training loss with increasing local epochs in this configuration, while measured accuracy remained 0.5000.


---

22. E8 — Data Quantity Imbalance

Objective

Evaluate federated training with unequal quantities of local training data while using balanced evaluation data.

Configuration:

Clients: 4
Rounds: 3
Training participation: 100%
Evaluation participation: 100%
Partitioning: IID
Local epochs: 5
Batch size: 4
Learning rate: 0.01
Optimizer: SGD
Seed: 42

Training distribution:

Client 0 -> 4 examples
Client 1 -> 8 examples
Client 2 -> 8 examples
Client 3 -> 12 examples

Total -> 32 examples

Evaluation distribution:

Client 0 -> 8 examples
Client 1 -> 8 examples
Client 2 -> 8 examples
Client 3 -> 8 examples

Total -> 32 examples

Results:

Round	Train Loss	Eval Loss	Accuracy	Eval Examples

1	0.680932	0.686110	0.5000	32
2	0.680199	0.685398	0.5000	32
3	0.679574	0.684794	0.5000	32


The experiment exercises the example-weighted aggregation path using heterogeneous local training-data quantities.


---

23. E9 — Centralized vs Federated Training

Objective

Compare centralized training with the existing federated workflow under matched training-data exposure.

Common configuration:

Training samples: 32
Batch size: 4
Learning rate: 0.01
Optimizer: SGD
Seed: 42

Both configurations process:

480 example-passes


---

E9-A — Centralized Training

Samples: 32
Epochs: 15
Batch size: 4
Example-passes: 480

Results:

Metric	Result

Epochs	15
Samples processed	480
Batches processed	120
Final train loss	0.6817503422
Evaluation samples	32
Evaluation loss	0.6809001043
Accuracy	0.5000
Runtime	16.54s


The centralized experiment was implemented using the existing:

Trainer
Evaluator
FlowerSmokeTestModel
FedMedDataset


---

E9-B — Federated Training

The federated baseline reuses the previously validated E7-C experiment.

Configuration:

Clients: 4
Training examples/client: 8
Rounds: 3
Local epochs: 5
Training example-passes: 480

Results:

Metric	Result

Federated rounds	3
Local epochs	5
Training examples/round	32
Final train loss	0.672117
Final evaluation loss	0.684518
Accuracy	0.5000
Runtime	311.00s


The federated runtime includes Flower orchestration and distributed-runtime overhead.

These results are specific to this experimental configuration and are not treated as general conclusions about centralized or federated learning.


---

24. Experiment Status

E1 — Baseline Reproducibility                 COMPLETED
E2 — Client Count                             COMPLETED
E3 — Training Participation Fraction          COMPLETED
E4 — Number of Federated Rounds               COMPLETED
E5 — IID vs Label Skew                        COMPLETED
E6 — Client Failure / Dropout                 COMPLETED
E7 — Local Epochs                             COMPLETED
E8 — Data Quantity Imbalance                  COMPLETED
E9 — Centralized vs Federated Training        COMPLETED


---

25. Upgrade and Refactoring Work

After completing the experimental phase, FedMed entered a module-upgrade phase.

The upgrade workflow is:

Inspect
   |
Understand
   |
Identify actual problem
   |
Design upgrade
   |
Implement
   |
Unit Tests
   |
Integration Tests
   |
Full Regression
   |
Commit

The first upgrade work focused on:

src/fl/orchestrator.py


---

26. Orchestrator Upgrade

The orchestrator was reviewed for unnecessary framework-specific data-loader construction.

Previously the orchestrator directly constructed:

torch.utils.data.DataLoader

The implementation was updated to use the existing FedMed abstraction:

src.data.loader.create_dataloader

This keeps data-loader construction behind the FedMed data layer.

The unused TensorDataset import was also removed.

The change was committed as:

0413414
refactor: use FedMed dataloader abstraction


---

27. Evaluation Split Correction

During the orchestrator review, an existing E8 integration mismatch was identified.

The partition loader already supported:

split="train"
split="eval"

but build_client() was previously constructing both loaders using the training split.

The implementation was corrected so that:

train_loader = self._create_partitioned_loader(
    partition_index
)

eval_loader = self._create_partitioned_loader(
    partition_index,
    split="eval",
)

Therefore the client now receives separate training and evaluation partitions.

A regression test was added to verify:

client_0 training samples = 4
client_0 evaluation samples = 8

The fix was committed as:

4804da3
fix: use evaluation split for client evaluation


---

28. Test Coverage

FedMed currently has a large automated test suite covering:

Configuration

Dataset behavior

Data loaders

Data partitioning

Models

Training

Evaluation

Metrics

Parameters

Federated clients

Strategies

Aggregation

Round coordination

Server behavior

Federation checkpoint persistence

Flower client integration

Flower server integration

Orchestrator composition

Failure handling

Centralized training experiment


Latest full regression:

609 passed
2 warnings

The warnings are third-party deprecation warnings from the installed Typer/Click environment.


---

29. Orchestrator Tests

The orchestrator now has direct tests covering:

Orchestrator construction
Multiple orchestrator construction
Strategy and aggregator composition
Torch thread configuration
Distinct train/evaluation partition loading
Client evaluation split usage

Current orchestrator test result:

6 passed


---

30. Current Repository Status

Current branch:

main

Latest commit:

4804da3
fix: use evaluation split for client evaluation

Previous commit:

0413414
refactor: use FedMed dataloader abstraction

Current test status:

609 passed
2 warnings


---

31. Running FedMed

Activate Environment

cd ~/fedmed
source .venv/bin/activate


---

Start Flower SuperLink

Terminal 1:

cd ~/fedmed
source .venv/bin/activate

flower-superlink --insecure

Current local APIs:

Control API : 9093
Runtime API : 9091
Fleet API   : 9092


---

Start SuperNode 1

Terminal 2:

cd ~/fedmed
source .venv/bin/activate

flower-supernode \
  --insecure \
  --superlink 127.0.0.1:9092 \
  --clientappio-api-address 0.0.0.0:9095 \
  --node-config "partition-id=0 num-partitions=4"


---

Start SuperNode 2

Terminal 3:

cd ~/fedmed
source .venv/bin/activate

flower-supernode \
  --insecure \
  --superlink 127.0.0.1:9092 \
  --clientappio-api-address 0.0.0.0:9096 \
  --node-config "partition-id=1 num-partitions=4"


---

Start SuperNode 3

Terminal 4:

cd ~/fedmed
source .venv/bin/activate

flower-supernode \
  --insecure \
  --superlink 127.0.0.1:9092 \
  --clientappio-api-address 0.0.0.0:9097 \
  --node-config "partition-id=2 num-partitions=4"


---

Start SuperNode 4

Terminal 5:

cd ~/fedmed
source .venv/bin/activate

flower-supernode \
  --insecure \
  --superlink 127.0.0.1:9092 \
  --clientappio-api-address 0.0.0.0:9098 \
  --node-config "partition-id=3 num-partitions=4"


---

Run the Flower Application

Terminal 6:

cd ~/fedmed
source .venv/bin/activate

FLWR_LOG_LEVEL=DEBUG flwr run . local-deployment --stream


---

32. Running Tests

Run the complete test suite:

cd ~/fedmed
.venv/bin/pytest -q

Expected current result:

609 passed, 2 warnings

Run orchestrator tests:

.venv/bin/pytest -q tests/test_orchestrator.py -s

Run data-loader tests:

.venv/bin/pytest -q tests/test_loader.py


---

33. Daily Development Check

Before committing changes:

cd ~/fedmed

.venv/bin/pytest -q

Check repository status:

git status --short

Review the latest commit:

git log -1 --oneline

The project follows:

Change
  |
Test
  |
Full Regression
  |
Review
  |
Commit


---

34. Configuration

The main project configuration is stored under:

configs/

Flower runtime configuration is managed separately through:

pyproject.toml

The architecture intentionally keeps:

FedMed application configuration

separate from:

Flower runtime configuration

The federation policy defined in [configs/config.yaml](configs/config.yaml)
is wired into the runtime by the composition root. Precedence and the
distinction between the two sources is described in the
"Federated Configuration Runtime Wiring" section below.


---

35. Technology Stack

Programming Language

Python 3.12

Machine Learning

PyTorch
NumPy

Federated Learning

Flower 1.34.0

Testing

pytest

Runtime Environment

WSL2 / Ubuntu

Version Control

Git
GitHub


---

36. Project Structure

FedMed/
│
├── app/
│   ├── client.py
│   ├── failure_mod.py
│   ├── main.py
│   └── server.py
│
├── configs/
│   └── config.yaml
│
├── src/
│   ├── aggregation/
│   │   └── fedavg.py
│   │
│   ├── common/
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   └── logging.py
│   │
│   ├── data/
│   │   ├── dataset.py
│   │   ├── loader.py
│   │   └── partitioner.py
│   │
│   ├── fl/
│   │   ├── aggregation.py
│   │   ├── checkpoint.py
│   │   ├── client.py
│   │   ├── observability.py
│   │   ├── orchestrator.py
│   │   ├── parameters.py
│   │   ├── rounds.py
│   │   ├── server.py
│   │   └── strategy.py
│   │
│   ├── models/
│   │   ├── base_model.py
│   │   └── model_factory.py
│   │
│   ├── monitoring/
│   │   ├── logger.py
│   │   └── metrics.py
│   │
│   └── training/
│       ├── evaluator.py
│       ├── metrics.py
│       └── trainer.py
│
├── tests/
│   ├── test_app_client.py
│   ├── test_app_server.py
│   ├── test_loader.py
│   ├── test_observability.py
│   ├── test_orchestrator.py
│   ├── test_partitioner.py
│   ├── test_e9_centralized.py
│   └── ...
│
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md


---

37. Current Development Status

Core Architecture                    COMPLETED
Training Infrastructure              COMPLETED
Evaluation Infrastructure            COMPLETED
FedAvg Aggregation                   COMPLETED
Federated Strategy                   COMPLETED
Flower Client Integration            COMPLETED
Flower Server Integration            COMPLETED
Multi-Node Flower Runtime            COMPLETED
IID Partitioning                     COMPLETED
Label-Skew Partitioning              COMPLETED
Client Failure Experiment            COMPLETED
Federated Experiments E1-E9          COMPLETED
Centralized Comparison               COMPLETED
Federation Checkpointing             COMPLETED
Orchestrator Upgrade                 IN PROGRESS
Automated Regression Suite            ACTIVE


---

38. Current Development Direction

FedMed has completed its initial federated-learning runtime and experimentation stage.

The project is now moving into the module upgrade and productionization stage.

The upgrade process will continue module-by-module:

Inspect
   ↓
Understand
   ↓
Identify actual issue
   ↓
Design focused improvement
   ↓
Implement
   ↓
Add/update tests
   ↓
Integration verification
   ↓
Full regression
   ↓
Commit

The goal is to improve the existing implementation without unnecessarily changing the established FedMed architecture.


---

39. Federation Checkpointing

`FederatedServer` owns the only long-lived cross-round state in FedMed:

- the current global `ParameterPayload`
- the most recently completed round number
- the immutable round history

Until now that state existed only in memory, so a crash mid-federation lost all progress and there was no way to resume.

The `checkpoint` block in `configs/config.yaml` was already declared and validated but had no consumer. It now has one.

New module

`src/fl/checkpoint.py` is framework-independent and Flower-free. It contains three public names:

- `CheckpointError` — a specific subclass of the existing `FederatedLearningError` hierarchy
- `FederationCheckpoint` — an immutable, self-describing snapshot of resumable state
- `FederationCheckpointStore` — atomic filesystem persistence

Self-describing checkpoints

A checkpoint stores the parameter values **and** the structural contract they were captured against (names, shapes, dtypes). That makes a mismatch between the saved federation and the current one detectable instead of silently producing a corrupt resume. Restore reuses the existing Phase 3.1 `validate_parameters()` and `validate_contract_compatibility()` boundaries rather than introducing a second validation mechanism.

Storage is a single uncompressed `.npz` archive holding the round number, the contract metadata, and one array per parameter. Because different parameters can have different ranks, shapes are stored in one flattened `int64` vector delimited by a per-parameter start offset.

Crash safety

`save()` serializes into a temporary file in the destination directory and then moves it into place with `os.replace`. An interrupted write therefore cannot truncate an existing checkpoint: the previous file stays intact and the temporary file is an ignorable artifact.

Persistence also never runs ahead of progress. The server saves only *after* a round is fully committed, so a failed round is never recorded as progress, and a failed save surfaces to the caller without rolling back a completed round.

Corruption is rejected loudly

A missing, truncated, or inconsistent checkpoint raises `CheckpointError` rather than falling back to the initial model. Silently restarting from scratch is the worst failure mode a resume feature can have, because it discards progress without telling anyone. The one exception is `load_latest()` on an empty store, which returns `None` because "nothing saved yet" is a normal state.

Usage

Attaching a store is optional and keyword-only. Omitting it preserves earlier behavior exactly: no filesystem access, no state written.

```python
store = FederationCheckpointStore.from_config(
    config.checkpoint
)

server = FederatedServer(
    clients=clients,
    strategy=strategy,
    coordinator=coordinator,
    initial_parameters=initial_parameters,
    checkpoint_store=store,
)

# Saved automatically after each committed round that
# falls on the store's save_every_round interval.
server.run(num_rounds=3)

# Or persist explicitly at any point.
server.save_checkpoint()

# After a restart, a fresh server resumes exactly where
# the previous one stopped.
resumed = FederatedServer(
    clients=clients,
    strategy=strategy,
    coordinator=coordinator,
    initial_parameters=initial_parameters,
    checkpoint_store=store,
)

resumed.restore_checkpoint()

# Continues at the next contiguous round number.
resumed.run_round()
```

`FederationCheckpointStore.from_config()` is what gives the existing `CheckpointConfig` section its first real consumer.

A restored server resumes with an empty round history. Durable state is the global model plus round progression; `RoundExecution` carries live client result objects and per-round metrics that are not part of the durable state.

Architecture fit

- Pure NumPy and filesystem work under `src/`; no Flower import.
- Reuses `ParameterPayload`, `copy_parameters`, `validate_parameters`, `ParameterContract`, `FederatedClient.parameter_contract`, and `CheckpointConfig`.
- Strategy and Aggregator are untouched.
- The four existing positional `FederatedServer` constructor arguments are unchanged, so all current callers and tests behave identically.

Current scope limit

The Flower runtime path in `app/server.py` adapts `FedAvgStrategy` directly to Flower and does not route through `FederatedServer`. Checkpointing is therefore available at the `FederatedServer` level only, and `checkpoint.enabled` does **not** yet affect a live Flower run. Wiring the Flower runtime through `FederatedServer` is a separate, larger upgrade and was deliberately not attempted here.

Checkpoint test coverage

88 tests covering construction and defensive copying, contract mismatch rejection, non-finite and malformed payload rejection, round-trip fidelity across mixed-rank shapes, round discovery, unrelated-file tolerance, corruption and incomplete-archive rejection, atomic-save failure preserving a good checkpoint, automatic save intervals, manual save/restore round-trips, resumed round contiguity, mismatched-federation rejection leaving state untouched, and failure not advancing the checkpoint.


---

40. Federated Configuration Runtime Wiring

Where federated configuration is defined

The framework-independent federated policy is defined in:

configs/config.yaml

under the `federated:` block:

- `num_rounds`
- `min_clients`
- `min_available_clients`
- `fraction_fit`
- `fraction_evaluate`
- `strategy`

It is parsed into `FedMedConfig.federated` (`FederatedConfig`) by
`src/common/config.py` and loaded by the composition root
`FedMedOrchestrator`.

What it controls

`FedMedOrchestrator.build_server_app()` forwards the federated policy into
the Flower ServerApp runtime:

    federated.num_rounds             -> ServerApp num_rounds
    federated.fraction_fit           -> Flower fraction-train baseline
    federated.fraction_evaluate      -> Flower fraction-evaluate baseline
    federated.min_available_clients  -> Flower min-available-nodes baseline

Before this wiring, `build_server_app()` hardcoded `num_rounds=1` and the
Flower adapter consumed only its own `run_config` values. Changing
`configs/config.yaml` therefore did not change runtime behavior. It now
does: the centralized configuration drives the runtime, and changing it
changes the number of rounds and the client-selection policy used by the
server.

The framework-independent `FederatedServer` continues to take an explicit
round count per invocation (`FederatedServer.run(num_rounds=...)`). Callers
that want config-driven behavior pass `config.federated.num_rounds`
explicitly; `FederatedServer` itself remains configuration-free.

Configuration precedence

Two sources configure the federated runtime, and precedence is explicit:

1. Flower `run_config` (`[tool.flwr.app.config]` in `pyproject.toml`) is
   the highest-precedence runtime override. For its keys
   (`num-server-rounds`, `fraction-train`, `fraction-evaluate`,
   `min-available-nodes`) it always wins.

2. The centralized `FederatedConfig` (`configs/config.yaml`) supplies the
   runtime baseline for `num_rounds`, `fraction_fit`, `fraction_evaluate`
   and `min_available_clients`.

3. The `app/server.py` adapter defaults (`num_rounds=1`, fractions `1.0`,
   `min_available_nodes=1`) apply only when neither source provides a value.

Each `run_config` key overrides only its matching baseline value; keys that
are absent fall back to the centralized configuration.

Framework-independent vs Flower-specific configuration

- `configs/config.yaml` is framework-independent. It drives the FedMed core
  and, through the orchestrator, provides the runtime baseline.
- `pyproject.toml` `[tool.flwr.app.config]` is Flower-specific runtime
  configuration. It is an independent override and is not generated from
  the centralized configuration.
- FedMed does not claim Flower automatically consumes
  `configs/config.yaml`. The value reaches the runtime because the
  orchestration root passes the `FederatedConfig` fields into
  `create_server_app()`, after which Flower's `run_config` remains an
  authoritative override. This is what keeps the E1-E9 experiments
  reproducible: their explicit `run_config` values keep taking effect.

Current scope limit

`federated.min_clients` has no consumer in the Flower adapter yet. It is a
reserved federation-policy field (a minimum *selected* client floor) that
the current `FedMedFlowerStrategy` does not represent. Only
`min_available_clients` is used, as the availability gate and selection
floor on the number of Flower nodes.


---

41. Structured Federated Round Observability

What was missing

After a federation finished, "what happened in round 3?" was answerable in
three places and no good one of them:

- `RoundResult` was structured but carried live `FederatedFitResult` /
  `FederatedEvaluateResult` objects, including each client's full parameter
  payload and per-client metric mappings.
- Round information was otherwise spread across coordinator return values
  and adapter-side logging, so it was not comparable across rounds.
- The Flower runtime emitted `MetricRecord` values that are not a FedMed
  record and carry no client-failure accounting.

None of those is a compact, serializable, per-round record. `src/fl/observability.py`
adds one without changing any existing abstraction.

New module

`src/fl/observability.py` is framework-independent and Flower-free. It exports:

- `FederatedRoundSummary` — frozen, flat record of one completed round
- `parameter_fingerprint()` — deterministic truncated SHA-256 digest of a payload
- `ACCURACY_METRIC_KEY` / `FINGERPRINT_LENGTH`

```python
from src.fl.observability import FederatedRoundSummary

summary = FederatedRoundSummary.from_round_execution(execution)

summary.round_number            # 3
summary.status                  # RoundState.COMPLETED
summary.successful_clients      # ('client_a', 'client_b')
summary.failed_client_ids       # ('client_c',)
summary.participation_count     # 2
summary.dropout_count           # 1
summary.training_examples       # 16
summary.training_loss           # 0.31
summary.evaluation_accuracy     # None when the round did not evaluate
summary.parameter_fingerprint   # '9f2c41ab7e0d5c38'
```

The server exposes the same records for its whole history:

```python
server.run(num_rounds=3)

for summary in server.round_summaries:
    print(summary.round_number, summary.dropout_count)

# or a single round
server.summary_for_round(2)
```

Failures stay separated, exactly as in `RoundResult`

`failed_clients` reuses the existing `ClientFailure` type and carries only
`TRAINING` failures; `evaluation_failures` carries only `EVALUATING` failures. A
client that trained successfully and then failed evaluation is still counted
as a successful training client. A client that failed training is never
reported as successful, so the E6 dropout shape is preserved exactly.

`src/fl/rounds.py` and `app/failure_mod.py` are untouched.

Reporting only, never policy

The summary is a projection. It cannot influence which clients are selected,
which parameters are aggregated, or whether a round succeeds:

- `training_examples` / `evaluation_examples` are plain sums over already
  recorded client results.
- `training_loss` and the evaluation observations are sample-weighted means
  used for reporting. The round's parameters were already produced by the
  Strategy/Aggregator before a summary exists.

If a runtime already aggregated evaluation through its Strategy, pass that
result in and it is recorded verbatim, so reporting matches the runtime
exactly:

```python
FederatedRoundSummary.from_round_execution(
    execution,
    evaluation=strategy.aggregate_evaluate(
        execution.result.evaluation_results, round_number
    ),
)
```

Parameter fingerprint

`parameter_fingerprint()` hashes each parameter's shape, dtype, and bytes, so
identical payloads always produce an identical 16-character digest and a
changed model produces a different one. That makes it a cheap "did the model
actually change?" signal across rounds. The Flower adapter's private
`_parameter_fingerprint` in `app/server.py` now delegates to the core helper
instead of keeping a second implementation.

Checkpoint relationship

A summary is derived on access from round history. It is never stored and is
never part of `FederationCheckpoint`, so a restored server resumes with empty
`round_summaries` exactly as it already resumes with empty `round_history`.
Checkpoint contents are unchanged.

Usage

```python
server.run(num_rounds=3, evaluate=True)

summary = server.summary_for_round(3)

if summary.dropout_count:
    # participation is already reported, no client lookup needed
    print(summary.successful_clients, summary.failed_client_ids)
```

`round_summaries` is recomputed on access rather than cached, so it is always
consistent with `round_history` and adds no server state.

Architecture fit

- Pure NumPy/dataclasses work under `src/`; no Flower import.
- Reuses `RoundExecution`, `RoundResult`, `ClientFailure`, `RoundState`, and
  `FedAvgEvaluationResult`. No existing type is extended or replaced.
- Strategy and Aggregator are untouched.
- `FederatedServer`'s four positional constructor arguments are unchanged; the
  additions are a read-only property and one lookup method.

Current scope limit

As with checkpointing, the Flower runtime path in `app/server.py` adapts
`FedAvgStrategy` directly and does not route through `FederatedServer`, so
`round_summaries` is available at the `FederatedServer` level only. The core
fingerprint helper is shared by both paths, so there is a single definition.

Round observability test coverage

88 tests covering valid summary creation, round-number validation,
inconsistent client-set rejection, successful/failed accounting, metric
validation including non-finite and out-of-range accuracy, the empty
failed-client set, `RoundExecution` derivation, `FederatedServer.round_summaries`
and `summary_for_round`, multiple completed rounds, existing server behavior
under dropout and total failure, checkpoint restore leaving summaries empty,
fingerprint determinism and sensitivity, and an AST scan proving the module
imports no Flower module.


---

About

FedMed is a privacy-preserving cross-silo federated learning engine for collaborative medical AI.

The project focuses on building a modular, testable and experimentally validated federated learning runtime that can support future medical machine-learning workloads without requiring centralized sharing of local training data.

### One thing I recommend before pasting

Your current GitHub README is **much more outdated than the actual repository**. The version above updates the important stale parts:

- `609 passed` instead of `577`
- E1–E9 included
- E6/E7/E8/E9 results included
- current orchestrator refactoring included
- evaluation-split correction included
- current Flower topology with **4 SuperNodes**
- `FedMedFlowerStrategy` described as the Flower adapter
- Strategy/Aggregator separation preserved
- current development status changed from runtime-completion to **module upgrade phase**
- latest commits `0413414` and `4804da3` included
