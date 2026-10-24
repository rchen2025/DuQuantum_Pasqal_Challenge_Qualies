# pasqal_cloud 0.24.0 — code-level API digest

Source: `/home/claude/vendor/pasqal_cloud/` (version `0.24.0`, `deprecation_date = '2027-09-08'`). pulser-core 1.9.1 is the companion.
pasqal_cloud imports fine with `PYTHONPATH=/home/claude/vendor` (needs `pydantic`, `requests`, `tenacity`, `jwt`, `pulser`).

## 0. Two layers — which one to use

1. **Low-level REST client**: `pasqal_cloud.pasqal_cloud_client.PasqalCloudClient` (the old `SDK` is now a thin deprecated alias in `pasqal_cloud/sdk.py`: `class SDK(PasqalCloudClient)` that emits `DeprecationWarning("'SDK' is an alias to 'PasqalCloudClient' …")`). Works with serialized sequences and returns `Batch`/`Job` pydantic models.
2. **Pulser-integrated layer** (new in this version, top-level exports): `pasqal_cloud.PasqalCloudConnection` (a `pulser.backend.remote.RemoteConnection`), `pasqal_cloud.OVHConnection`, and `pasqal_cloud.backends.{RemoteMPSBackend, RemoteSVBackend, RemoteEmuFreeBackend}` (RemoteBackend + EmulatorBackend). These take a `pulser.Sequence` and return `pulser.backend.remote.RemoteResults`.

`pasqal_cloud/__init__.py.__all__ = ["PasqalCloudConnection", "OVHConnection", "RemoteMPSBackend", "RemoteSVBackend", "RemoteEmuFreeBackend"]`. Every other historical root-level name (`SDK`, `Batch`, `Job`, `EmulatorType`, `DeviceTypeName`, `BatchStatus`, `JobStatus`, `TokenProvider`, `Endpoints`, errors, filters, ...) is still importable from `pasqal_cloud` via `__getattr__` but emits `DeprecationWarning` pointing to the canonical module (`pasqal_cloud._deprecations._DEPRECATED_EXPORTS`). Canonical imports:

```python
from pasqal_cloud.pasqal_cloud_client import PasqalCloudClient
from pasqal_cloud.sdk import SDK                      # deprecated alias
from pasqal_cloud.batch import Batch, RESULT_POLLING_INTERVAL
from pasqal_cloud.job import Job, CreateJob
from pasqal_cloud.device import DeviceTypeName, EmulatorType, BaseConfig, EmuFreeConfig, EmuTNConfig
from pasqal_cloud.device.configuration import EmuFresnelConfig, InvalidConfiguration
from pasqal_cloud.utils.constants import BatchStatus, JobStatus, QueuePriority
from pasqal_cloud.utils.filters import BatchFilters, JobFilters, CancelJobFilters, RebatchFilters, PaginationParams
from pasqal_cloud.utils.responses import PaginatedResponse, BatchCancellationResponse, JobCancellationResponse
from pasqal_cloud.endpoints import Endpoints, Auth0Conf, TokenProviderConf, Region
from pasqal_cloud.authentication import TokenProvider, ExpiringTokenProvider, Auth0TokenProvider, PasswordGrantTokenProvider
from pasqal_cloud import errors
```

## 1. `PasqalCloudClient` (= `SDK`) constructor

```python
PasqalCloudClient(
    username: Optional[str] = None,
    password: Optional[str] = None,
    token_provider: Optional[TokenProvider] = None,
    endpoints: Optional[Endpoints] = None,
    auth0: Optional[Auth0Conf] = None,            # deprecated -> use auth_config
    webhook: Optional[str] = None,                # URL where job results are POSTed
    project_id: Optional[str] = None,
    client_class: Type[HTTPClient] = HTTPClient,
    region: Optional[Region] = None,              # Region = Literal["fr", "sa"]; "sa" -> KSA endpoints + Keycloak
    auth_config: Optional[TokenProviderConf] = None,
)
```
* Auth options (docstring): no args → unauthenticated public features only; `username`+`password`; `username` only → prompts for password (`getpass`); custom `token_provider` (must implement `get_token() -> str`).
* `auth0` and `auth_config` together → `ValueError`. `project_id` is required for batch creation (`HTTPClient.project_id` raises `ValueError("You need to set a project_id.")`).
* `_check_sdk_version()` warns 30 days before and after `deprecation_date`.
* Attributes: `.batches: dict[str, Batch]`, `.workloads`, `.webhook`, `.project_id` (property with setter), `.user_token()`.
* Endpoints: `Endpoints(core="https://apis.pasqal.cloud/core-fast", account="https://apis.pasqal.cloud/account")`; `Endpoints.from_region("sa")` → `apis.sa.pasqal.cloud`. Auth0 prod: domain `authenticate.pasqal.cloud`, token endpoint `https://pasqal.eu.auth0.com/oauth/token`, realm `pcs-users`.

### `create_batch`

```python
PasqalCloudClient.create_batch(
    self,
    serialized_sequence: Optional[str],          # seq.to_abstract_repr(); None -> every job must carry its own serialized_sequence
    jobs: List[CreateJob],                        # [{"runs": int, "variables": dict|None, "serialized_sequence": str|None}]
    complete: Optional[bool] = None,              # deprecated (opposite of open)
    open: Optional[bool] = None,                  # True -> batch stays open; add with Batch.add_jobs, finish with Batch.close()
    emulator: Optional[EmulatorType] = None,      # deprecated -> device_type; both set -> InvalidDeviceTypeSet
    device_type: Optional[DeviceTypeName] = None, # None -> device stored in the serialized sequence (i.e. QPU)
    configuration: Optional[BaseConfig] = None,   # EmuTNConfig / EmuFreeConfig; sent as "configuration": conf.to_dict()
    backend_configuration: Optional[str] = None,  # EmulationConfig.to_abstract_repr() JSON string (EMU_MPS / EMU_SV / EMU_FREE)
    wait: bool = False,                           # block until no PENDING/RUNNING jobs, then batch.refresh()
    fetch_results: bool = False,                  # deprecated alias of wait
    tags: Optional[list[str]] = None,
) -> Batch
```
* Payload sent: `{"sequence_builder", "webhook", "jobs", "open", "device_type", ["configuration"], ["tags"], ["backend_configuration"], "project_id"}`.
* Validation errors (all `BaseException` subclasses in `pasqal_cloud.errors`): `InvalidDeviceTypeSet`, `OnlyCompleteOrOpenCanBeSet`, `MissingAllJobSequence`, `MissingJobSequence(indexes)`, `JobSequenceVariablesConflict` (a job with both `serialized_sequence` and `variables`). `EMU_TN` emits a DeprecationWarning ("Use EMU_MPS instead"). HTTP failure → `BatchCreationError`.
* `CreateJob(TypedDict, total=False)`: `runs: int | None`, `variables: dict | None`, `serialized_sequence: Optional[str]` (renamed to `"sequence"` in the API payload by `create_jobs_to_api_payload`).

### Other client methods

```python
get_batch(id: str, fetch_results: bool = False) -> Batch           # fetch_results deprecated/no-op
get_batch_status(id: str) -> BatchStatus
get_batches(filters: BatchFilters | None = None, pagination_params: PaginationParams | None = None) -> PaginatedResponse   # .total, .offset, .results: list[Batch]; newest first; default limit 100
cancel_batch(id: str) -> Batch
cancel_batches(batch_ids: List[str]) -> BatchCancellationResponse  # .batches, .errors
rebatch(id: UUID | str, filters: RebatchFilters | None = None) -> Batch
add_jobs(batch_id: str, jobs: List[CreateJob], wait: bool = False) -> Batch
close_batch(batch_id: str) -> Batch                                # complete_batch() is the deprecated alias
get_job(id: str, wait: bool = False) -> Job
get_job_status(id: str) -> JobStatus
get_jobs(filters: JobFilters | None = None, pagination_params: PaginationParams | None = None) -> PaginatedResponse
cancel_job(id: str) -> Job
cancel_jobs(batch_id: UUID | str, filters: CancelJobFilters | None = None) -> JobCancellationResponse
create_workload(workload_type: str, backend: str, config: dict, wait: bool = False) -> Workload
get_workload(id: str, wait: bool = False) -> Workload
cancel_workload(id: str) -> Workload
get_device_specs_dict() -> Dict[str, str]          # {device_name: pulser abstract-repr JSON}; falls back to public specs when unauthenticated
set_batch_tags(batch_id: str, tags: list[str]) -> Batch
get_all_projects() -> list[Project]
switch_to_project(project_id: str) -> str          # ProjectNotFoundError if not a member
```

Filters (pydantic): `BatchFilters(id, start_date, end_date, project_id, user_id, device_type, status: BatchStatus|list, open: bool, queue_priority, tag: str)`; `JobFilters(id, start_date, end_date, min_runs, max_runs, project_id, user_id, batch_id, status: JobStatus|list, errors: bool)`; `PaginationParams(offset: int = 0, limit: int = 100 (1..100))`.

## 2. Enums (verbatim)

```python
class DeviceTypeName(StrEnum):            # pasqal_cloud/device/device_types.py
    EMU_FREE = "EMU_FREE"
    EMU_TN = "EMU_TN"
    EMU_FRESNEL = "EMU_FRESNEL"
    EMU_MPS = "EMU_MPS"
    EMU_SV = "EMU_SV"
    FRESNEL = "FRESNEL"
    FRESNEL_CAN1 = "FRESNEL_CAN1"

class EmulatorType(StrEnum):              # pasqal_cloud/device/emulator_types.py  — DEPRECATED, use DeviceTypeName
    EMU_FREE = "EMU_FREE"
    EMU_TN = "EMU_TN"
    EMU_FRESNEL = "EMU_FRESNEL"
    EMU_MPS = "EMU_MPS"

class BatchStatus(Enum):                  # pasqal_cloud/utils/constants.py
    PENDING = "PENDING"; RUNNING = "RUNNING"; DONE = "DONE"; ERROR = "ERROR"; CANCELED = "CANCELED"; TIMED_OUT = "TIMED_OUT"
class JobStatus(Enum):
    PENDING = "PENDING"; RUNNING = "RUNNING"; DONE = "DONE"; ERROR = "ERROR"; CANCELED = "CANCELED"
class QueuePriority(Enum):  LOW = "LOW"; MEDIUM = "MEDIUM"; HIGH = "HIGH"
class ResultType(StrEnum):  COUNTER = "counter"; RUN = "run"; SAMPLE = "sample"; EXPECTATION = "expectation"
```
Note: the `Batch.status` / `Job.status` fields are plain `str` on the models; docstrings list `PENDING, RUNNING, DONE, CANCELED, TIMED_OUT, ERROR, PAUSED` (PAUSED is not in the Enum).

## 3. Device configuration dataclasses (`pasqal_cloud/device/configuration/`)

```python
@dataclass
class BaseConfig:
    strict_validation: bool = False               # validate emulator job as if for FRESNEL
    extra_config: Optional[Dict[str, Any]] = None # merged flat into to_dict(); may not override a dataclass field
    result_types: Optional[List[ResultType]] = None
    def to_dict(self) -> dict; @classmethod from_dict(conf) -> BaseConfig; allowed_result_types -> [ResultType.COUNTER]

@dataclass
class EmuFreeConfig(BaseConfig):  with_noise: bool = False
@dataclass
class EmuTNConfig(BaseConfig):    dt: float = 10.0; precision: str = "normal" ("low"|"normal"|"high"); max_bond_dim: int = 500
@dataclass
class EmuFresnelConfig:           # empty
```
For `EMU_MPS` / `EMU_SV` / `EMU_FRESNEL`, `Batch._load_configuration` returns `None` for `configuration`; those backends are configured through `backend_configuration` (a pulser `EmulationConfig` abstract repr). `RemoteMPSBackend.default_config = EmulationConfig(observables=[BitStrings()], num_gpus_to_use=1, autosave_dt=float("inf"), optimize_qubit_ordering=True)`; `RemoteSVBackend.default_config = EmulationConfig(observables=[BitStrings()])`.

## 4. `Batch` (`pasqal_cloud/batch.py`, pydantic, `extra="allow"`)

Fields: `open: bool`, `complete: bool`, `created_at: str`, `updated_at: str`, `device_type: str`, `project_id: str`, `id: str`, `user_id: str`, `status: str`, `jobs_count: int = 0`, `jobs_count_per_status: Dict[str, int] = {}`, `webhook`, `start_datetime`, `end_datetime`, `device_status`, `parent_id`, `configuration: BaseConfig | dict | None`, `backend_configuration: Optional[str]`, `tags: Optional[list[str]]`.
Properties: `sequence_builder -> Optional[str]` (lazy GET), `ordered_jobs -> List[Job]` (lazy GET of all pages, ordered by creation), `jobs -> dict[id, Job]` (deprecated).
Methods:
```python
add_jobs(jobs: List[CreateJob], wait: bool = False) -> None    # batch must be open
retry(job: Job, wait: bool = False) -> None
close(wait: bool = False, fetch_results: bool = False) -> None  # declare_complete() is the deprecated alias
cancel() -> None
refresh() -> None                                                # re-GET and update in place
set_tags(tags: list[str]) -> None
```
`RESULT_POLLING_INTERVAL = 2` (seconds; only used by `wait_for_workload`).

## 5. `Job` (`pasqal_cloud/job.py`)

Fields: `runs: int | None`, `batch_id: str`, `id: str`, `project_id: str`, `status: str`, `created_at`, `updated_at`, `errors: Optional[List[str]]`, `start_timestamp`, `end_timestamp`, `variables: Optional[Dict[str, Any]]`, `parent_id`.
Properties:
* `sequence -> Optional[str]` (lazy).
* `full_result -> Optional[JobResult]` — lazy `GET …/jobs/{id}/results_link` then download; `JobResult(TypedDict)`: `raw: List[str]`, `counter: Dict[str, Any]`, `serialised_results: Optional[str]` (a pulser `Results` abstract repr, present for EMU_MPS/EMU_SV when the backend returns full observables).
* `result -> Optional[Dict[str, Any]]` = `full_result["counter"]` — bitstring → count. (Bit order follows the sequence register; `'1'` = Rydberg in the ground-rydberg basis.)
Methods: `cancel() -> dict`.

## 6. Polling / waiting (`pasqal_cloud/utils/polling.py`)

```python
POLL_WAIT_INITIAL_SECONDS = 2
POLL_WAIT_MAX_SECONDS = 30
NON_TERMINAL_STATUSES = frozenset({"PENDING", "RUNNING"})
def poll_status(fetcher: Callable[[], T], should_continue: Callable[[T], bool]) -> T   # tenacity wait_exponential(multiplier=2, max=30), stop_never
def batch_still_running(status_data: dict) -> bool   # any(jobs_count_per_status[s] > 0 for s in {"PENDING","RUNNING"})
def job_still_running(status_data: dict) -> bool     # status in {"PENDING","RUNNING"}
```
Schedule: 2s, 4s, 8s, 16s, 30s, 30s, … with **no total timeout** — `wait=True` blocks until terminal (or until the fetch raises). `create_batch(wait=True)` / `add_jobs(wait=True)` / `Batch.close(wait=True)` poll the batch-status endpoint; `get_job(id, wait=True)` polls the job-status endpoint. Retries on HTTP errors are handled separately by `pasqal_cloud.utils.retry` on the HTTP client; the `PasqalCloudConnection` layer additionally wraps `create_batch`/`add_jobs`/`get_batch` in `tenacity.retry(wait_exponential(multiplier=1, max=60), stop_after_attempt(5))`.

## 7. Errors (`pasqal_cloud/errors.py`)

All HTTP-derived errors subclass `ExceptionWithResponseContext(BaseException)` and carry the response body in the message: `BatchCreationError`, `BatchFetchingError`, `BatchCancellingError`, `BatchClosingError`, `RebatchError`, `BatchSetTagsError`, `JobCreationError`, `JobFetchingError`, `JobCancellingError`, `JobRetryError` (plain Exception), `WorkloadFetchingError`, `WorkloadCreationError`, `WorkloadCancellingError`, `DeviceSpecsFetchingError`, `ProjectFetchingError`, `ProjectNotFoundError(project_id)`, plus the validation errors listed in §1. Note these inherit from `BaseException`, so `except Exception` will NOT catch them — catch `errors.BatchException` / `errors.JobException` or `BaseException`.

## 8. Authentication (`pasqal_cloud/authentication.py`)

`TokenProvider(ABC).get_token() -> str`; `ExpiringTokenProvider` caches with `expiry_window = 1 min` and requires `_query_token() -> {"access_token": ..., "expires_in"?: ...}`; `PasswordGrantTokenProvider(username, password, config: TokenProviderConf, verify=True)`; `Auth0TokenProvider(username, password, auth0: Auth0Conf)`; `InsecureAuth0TokenProvider` (SSL off, tests only). `HTTPBearerAuthenticator` adds `Authorization: Bearer <token>`.

## 9. Pulser-integrated layer

### `PasqalCloudConnection(RemoteConnection)` (`pasqal_cloud/pasqal_cloud_connection.py`)

```python
PasqalCloudConnection(username: str = "", password: str = "", project_id: str = "", region: Region = "fr", **kwargs)  # kwargs -> PasqalCloudClient
.cloud_client: PasqalCloudClient
submit(sequence: Sequence, wait=False, open=False, batch_id=None, device_type: DeviceTypeName | None = None,
       backend_configuration: EmulationConfig | None = None, **kwargs) -> RemoteResults   # kwargs["job_params"] -> jobs
fetch_available_devices() -> dict[str, pulser.devices.Device]   # deserialized get_device_specs_dict()
get_results(batch_id: str, job_ids: list[str] | None = None) -> RemoteResults
supports_open_batch() -> True
```
`_query_job_progress` turns each job into `(JobStatus[job.status], Results)` — using `Results.from_abstract_repr(full_result["serialised_results"])` when present, else `SampledResult(atom_order=register qubit_ids, meas_basis=..., bitstring_counts=job.result)`. `submit` adds a measurement to the sequence if missing (single-basis only).

`OVHConnection(**kwargs)` — same, but token from env var `PASQAL_PULSER_ACCESS_TOKEN` and `OvhClient` endpoints under `/api/v1/third-party-access/ovh`; raises `MissingEnvironmentVariableError`.

### `pasqal_cloud.backends`

```python
class RemoteEmulatorBackend(RemoteBackend, EmulatorBackend):
    def __init__(self, sequence, connection: RemoteConnection, *, config: EmulationConfig | None = None, mimic_qpu: bool = False)
    def run(self, job_params: list[JobParams] | None = None, wait: bool = False) -> RemoteResults   # job_params None -> [{}] (one job)
class RemoteMPSBackend(RemoteEmulatorBackend):     _device_type = DeviceTypeName.EMU_MPS
class RemoteSVBackend(RemoteEmulatorBackend):      _device_type = DeviceTypeName.EMU_SV
class RemoteEmuFreeBackend(RemoteEmulatorBackend): _device_type = DeviceTypeName.EMU_FREE
```
The config is serialized with `to_abstract_repr()` and sent as `backend_configuration`. Because these subclass the generic `EmulationConfig`, pass emulator-specific fields as plain kwargs (`EmulationConfig(observables=[BitStrings(num_shots=100)], dt=10, max_bond_dim=500, precision=1e-6)`); unknown keys are accepted (`_enforce_expected_kwargs=False`) and validated server-side.

**pulser-pasqal**: not installed in `/home/claude/vendor` (no `pulser_pasqal` directory; `pip show pulser-pasqal` → not found). Its role (`PasqalCloud` connection + `EmuTNBackend`/`EmuFreeBackend`) is superseded here by `pasqal_cloud.PasqalCloudConnection` and `pasqal_cloud.backends.*`. pulser-core's `pulser/backend/remote.py` only holds the abstract `RemoteConnection`/`RemoteBackend`/`RemoteResults` (see emulators.md §5).

## 10. Minimal snippets

### A. Low-level: submit a serialized sequence to EMU_MPS with 100 runs, poll, read the counter

```python
from pasqal_cloud.pasqal_cloud_client import PasqalCloudClient   # `from pasqal_cloud.sdk import SDK` is the deprecated alias
from pasqal_cloud.device import DeviceTypeName
from pasqal_cloud.utils.polling import poll_status, batch_still_running
from pulser.backend import EmulationConfig, BitStrings

client = PasqalCloudClient(username=USER, password=PASSWORD, project_id=PROJECT_ID)   # region="sa" for KSA

seq.measure("ground-rydberg")                   # PasqalCloudClient does NOT add a measurement for you
serialized = seq.to_abstract_repr()

mps_cfg = EmulationConfig(observables=[BitStrings(num_shots=100)], dt=10, precision=1e-6, max_bond_dim=500)

batch = client.create_batch(
    serialized_sequence=serialized,
    jobs=[{"runs": 100}],                       # CreateJob; add "variables": {...} for parametrized sequences
    device_type=DeviceTypeName.EMU_MPS,
    backend_configuration=mps_cfg.to_abstract_repr(),
    wait=False,
)

# poll (2s,4s,8s,16s,30s,30s,... until no PENDING/RUNNING jobs)
poll_status(fetcher=lambda: client._client.get_batch_status(batch.id), should_continue=batch_still_running)
batch.refresh()                                   # or: batch = client.get_batch(batch.id)
# (equivalently just pass wait=True to create_batch)

job = batch.ordered_jobs[0]
assert job.status == "DONE", job.errors
counts: dict[str, int] = job.result               # == job.full_result["counter"]
```

### B. Pulser-integrated: same thing via RemoteMPSBackend

```python
from pasqal_cloud import PasqalCloudConnection, RemoteMPSBackend
from pulser.backend import EmulationConfig, BitStrings

conn = PasqalCloudConnection(username=USER, password=PASSWORD, project_id=PROJECT_ID)
cfg = EmulationConfig(observables=[BitStrings(num_shots=100)], dt=10, max_bond_dim=500)
remote = RemoteMPSBackend(seq, connection=conn, config=cfg).run(wait=True)   # RemoteResults
counts = dict(remote[0].final_bitstrings)        # remote.results[0].get_result("bitstrings", 1.0)
# non-blocking: remote.get_batch_status(); remote.get_available_results()
```

### C. QPU

```python
from pulser.backend import QPUBackend, BackendConfig
dev = conn.fetch_available_devices()["FRESNEL"]           # rebuild seq on this device
res = QPUBackend(seq, conn, config=BackendConfig(default_num_shots=500)).run(job_params=[{"runs": 500}], wait=True)
counts = dict(res[0].final_bitstrings)
```
