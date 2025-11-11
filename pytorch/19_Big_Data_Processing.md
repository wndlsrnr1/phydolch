# 19. 빅데이터 처리 (Big Data Processing)

이번 단원에서는 **메모리에 올릴 수 없는 대용량 데이터를 PyTorch로 처리하는 방법**을 배워보겠습니다.

## 학습 목표
- 대용량 데이터의 문제점
- 청킹 (Chunking) 전략
- 메모리 맵 파일 (Memory-mapped files)
- 스트리밍 데이터 처리
- 분산 학습 기초
- 백엔드 개발자를 위한 최적화

## 왜 이 단원이 필요한가?

**실제 데이터는 항상 크다**:

- **이미지**: 수백만 장 (수 TB)
- **로그**: 일일 수 GB
- **센서 데이터**: 실시간 스트림
- **텍스트**: 수억 개 문서

백엔드 개발자라면 **스트리밍, 배치 처리, 분산 시스템**에 익숙할 것입니다. 이 경험을 딥러닝에 적용해봅시다!

---
# 1. 대용량 데이터의 문제
---

## 1.1 메모리 제약

### 문제 시나리오
```python
# ❌ 메모리 부족!
data = np.load('huge_dataset.npy')  # 100GB
# MemoryError: Unable to allocate ...
```

### 메모리 계산
```
이미지 1장: 224 × 224 × 3 × 4 bytes (float32) = 600KB
100만 장: 600KB × 1M = 600GB
```

## 1.2 해결 전략

1. **청킹 (Chunking)**: 데이터를 작은 조각으로
2. **메모리 맵**: 디스크를 메모리처럼 사용
3. **스트리밍**: 필요할 때만 로드
4. **분산 처리**: 여러 머신에 분산
5. **압축**: 저장 공간 절약

---
# 2. 청킹 (Chunking)
---

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

# 1. HDF5로 대용량 데이터 저장
def create_large_dataset(filename, num_samples=1000000):
    """
    HDF5 형식으로 대용량 데이터 생성
    """
    chunk_size = 10000
    
    with h5py.File(filename, 'w') as f:
        # 데이터셋 생성 (청크 단위로 저장)
        dset = f.create_dataset(
            'images',
            shape=(num_samples, 28, 28),
            dtype='float32',
            chunks=(chunk_size, 28, 28)  # 청크 크기
        )
        
        labels = f.create_dataset(
            'labels',
            shape=(num_samples,),
            dtype='int64',
            chunks=(chunk_size,)
        )
        
        # 청크 단위로 데이터 쓰기
        for i in range(0, num_samples, chunk_size):
            end = min(i + chunk_size, num_samples)
            dset[i:end] = np.random.randn(end - i, 28, 28)
            labels[i:end] = np.random.randint(0, 10, end - i)
            
            if (i // chunk_size) % 10 == 0:
                print(f"Progress: {i}/{num_samples}")
    
    print(f"✓ Dataset created: {filename}")

# 2. HDF5 Dataset 클래스
class HDF5Dataset(Dataset):
    """
    HDF5 파일을 읽는 Dataset
    메모리에 전체 데이터를 올리지 않음
    """
    def __init__(self, filename):
        self.filename = filename
        
        # 파일 열기 (읽기 전용)
        self.file = h5py.File(filename, 'r')
        self.images = self.file['images']
        self.labels = self.file['labels']
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 필요한 데이터만 읽기
        image = torch.FloatTensor(self.images[idx])
        label = torch.LongTensor([self.labels[idx]])
        return image, label
    
    def __del__(self):
        # 파일 닫기
        if hasattr(self, 'file'):
            self.file.close()

print("=== HDF5 청킹 전략 ===")
print("\n장점:")
print("  ✓ 메모리 효율적 (필요한 것만 로드)")
print("  ✓ 빠른 랜덤 액세스")
print("  ✓ 압축 지원")
print("\n사용법:")
print("  create_large_dataset('data.h5', num_samples=1000000)")
print("  dataset = HDF5Dataset('data.h5')")
print("  loader = DataLoader(dataset, batch_size=32)")
```

---
# 3. 메모리 맵 파일
---

```python
import numpy as np
import torch
from torch.utils.data import Dataset

# 1. NumPy Memory-mapped 파일 생성
def create_memmap_dataset(filename, num_samples=1000000):
    """
    Memory-mapped 배열 생성
    """
    # Memory-mapped 파일 생성
    images = np.memmap(
        filename + '_images.dat',
        dtype='float32',
        mode='w+',
        shape=(num_samples, 28, 28)
    )
    
    labels = np.memmap(
        filename + '_labels.dat',
        dtype='int64',
        mode='w+',
        shape=(num_samples,)
    )
    
    # 데이터 쓰기 (청크 단위)
    chunk_size = 10000
    for i in range(0, num_samples, chunk_size):
        end = min(i + chunk_size, num_samples)
        images[i:end] = np.random.randn(end - i, 28, 28)
        labels[i:end] = np.random.randint(0, 10, end - i)
        
        if (i // chunk_size) % 10 == 0:
            print(f"Progress: {i}/{num_samples}")
    
    # 디스크에 플러시
    images.flush()
    labels.flush()
    
    print(f"✓ Memmap dataset created: {filename}")

# 2. Memory-mapped Dataset
class MemmapDataset(Dataset):
    """
    Memory-mapped 파일을 읽는 Dataset
    """
    def __init__(self, filename, num_samples):
        # Memory-mapped 파일 열기 (읽기 전용)
        self.images = np.memmap(
            filename + '_images.dat',
            dtype='float32',
            mode='r',
            shape=(num_samples, 28, 28)
        )
        
        self.labels = np.memmap(
            filename + '_labels.dat',
            dtype='int64',
            mode='r',
            shape=(num_samples,)
        )
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # OS가 자동으로 페이징 처리
        image = torch.FloatTensor(self.images[idx])
        label = torch.LongTensor([self.labels[idx]])
        return image, label

print("=== Memory-mapped 파일 ===")
print("\n장점:")
print("  ✓ 매우 빠른 랜덤 액세스")
print("  ✓ OS 페이징 활용")
print("  ✓ 간단한 구현")
print("\n단점:")
print("  ✗ 압축 불가")
print("  ✗ 큰 디스크 공간 필요")
print("\nHDF5 vs Memmap:")
print("  - HDF5: 압축, 메타데이터, 복잡한 구조")
print("  - Memmap: 빠름, 간단, 원시 데이터")
```

---
# 4. 스트리밍 데이터 처리
---

```python
import torch
from torch.utils.data import IterableDataset, DataLoader
import os
from pathlib import Path

# 1. Iterable Dataset (스트리밍)
class StreamingDataset(IterableDataset):
    """
    파일을 순차적으로 읽는 스트리밍 Dataset
    """
    def __init__(self, data_dir, file_pattern='*.jpg'):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob(file_pattern))
    
    def __iter__(self):
        """
        파일을 하나씩 읽어서 yield
        """
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single-process
            iter_start = 0
            iter_end = len(self.files)
        else:
            # Multi-process: 파일을 worker에 분배
            per_worker = len(self.files) // worker_info.num_workers
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker
        
        # 파일 순회
        for file_path in self.files[iter_start:iter_end]:
            # 파일 읽기 (예: 이미지)
            # image = Image.open(file_path)
            # image = transform(image)
            
            # 여기서는 더미 데이터
            data = torch.randn(3, 224, 224)
            label = torch.LongTensor([0])
            
            yield data, label

# 2. 온라인 데이터 스트림
class OnlineStreamDataset(IterableDataset):
    """
    API나 큐에서 실시간 데이터 수신
    """
    def __init__(self, data_source):
        self.data_source = data_source
    
    def __iter__(self):
        while True:
            # 데이터 소스에서 가져오기
            # data = self.data_source.get()  # 예: 큐, API
            
            # 더미 데이터
            data = torch.randn(10)
            label = torch.LongTensor([0])
            
            yield data, label

# 3. 배치 단위 스트리밍
def stream_batches(data_source, batch_size=32):
    """
    데이터를 배치 단위로 스트리밍
    """
    batch_data = []
    batch_labels = []
    
    for data, label in data_source:
        batch_data.append(data)
        batch_labels.append(label)
        
        if len(batch_data) >= batch_size:
            yield (
                torch.stack(batch_data),
                torch.cat(batch_labels)
            )
            batch_data = []
            batch_labels = []
    
    # 남은 데이터
    if batch_data:
        yield (
            torch.stack(batch_data),
            torch.cat(batch_labels)
        )

print("=== 스트리밍 데이터 처리 ===")
print("\n사용 시나리오:")
print("  1. 실시간 로그 분석")
print("  2. 센서 데이터 처리")
print("  3. 온라인 학습")
print("  4. 매우 큰 데이터셋 (TB 단위)")
print("\n장점:")
print("  ✓ 메모리 사용량 일정")
print("  ✓ 무한 데이터 스트림 처리 가능")
print("  ✓ 실시간 처리")
```

---
# 5. 분산 학습 기초
---

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 1. DataParallel (단일 머신, 여러 GPU)
def train_with_dataparallel(model, dataloader, device):
    """
    DataParallel 사용 (간단하지만 느림)
    """
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # 학습 루프
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # ...

# 2. DistributedDataParallel (권장)
def setup_ddp(rank, world_size):
    """
    DDP 초기화
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 프로세스 그룹 초기화
    dist.init_process_group(
        backend='nccl',  # GPU용
        rank=rank,
        world_size=world_size
    )

def cleanup_ddp():
    dist.destroy_process_group()

def train_ddp(rank, world_size, model, dataset):
    """
    DDP 학습
    """
    # 초기화
    setup_ddp(rank, world_size)
    
    # 모델을 DDP로 래핑
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # DistributedSampler 사용
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler
    )
    
    # 학습 루프
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # 중요!
        
        for data, target in dataloader:
            data, target = data.to(rank), target.to(rank)
            output = ddp_model(data)
            # ...
    
    cleanup_ddp()

# 3. 실행
def main():
    world_size = torch.cuda.device_count()
    
    # 멀티프로세싱으로 각 GPU에 프로세스 생성
    torch.multiprocessing.spawn(
        train_ddp,
        args=(world_size, model, dataset),
        nprocs=world_size,
        join=True
    )

print("=== 분산 학습 ===")
print("\nDataParallel vs DDP:")
print("\nDataParallel:")
print("  - 간단함")
print("  - 단일 프로세스")
print("  - GPU 0에 병목")
print("\nDistributedDataParallel (DDP):")
print("  - 복잡함")
print("  - 멀티 프로세스")
print("  - 효율적 (권장)")
print("  - 여러 머신 지원")
```

---
# 6. 백엔드 개발자를 위한 최적화
---

```python
import torch
from torch.utils.data import DataLoader
import time
from concurrent.futures import ThreadPoolExecutor

# 1. Prefetching
class PrefetchLoader:
    """
    다음 배치를 미리 로드
    """
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
    
    def __iter__(self):
        loader_iter = iter(self.loader)
        self.preload(loader_iter)
        
        while self.next_data is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            data = self.next_data
            self.preload(loader_iter)
            yield data
    
    def preload(self, loader_iter):
        try:
            self.next_data = next(loader_iter)
        except StopIteration:
            self.next_data = None
            return
        
        with torch.cuda.stream(self.stream):
            self.next_data = [
                x.to(self.device, non_blocking=True)
                for x in self.next_data
            ]

# 2. 캐싱
class CachedDataset:
    """
    자주 사용하는 데이터를 메모리에 캐싱
    """
    def __init__(self, dataset, cache_size=1000):
        self.dataset = dataset
        self.cache = {}
        self.cache_size = cache_size
        self.access_count = {}
    
    def __getitem__(self, idx):
        # 캐시에 있으면 반환
        if idx in self.cache:
            self.access_count[idx] += 1
            return self.cache[idx]
        
        # 캐시에 없으면 로드
        data = self.dataset[idx]
        
        # 캐시 공간이 있으면 저장
        if len(self.cache) < self.cache_size:
            self.cache[idx] = data
            self.access_count[idx] = 1
        else:
            # LRU: 가장 적게 사용된 항목 제거
            lru_idx = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_idx]
            del self.access_count[lru_idx]
            self.cache[idx] = data
            self.access_count[idx] = 1
        
        return data

# 3. 비동기 데이터 로딩
class AsyncDataLoader:
    """
    백그라운드에서 데이터 로드
    """
    def __init__(self, dataset, batch_size, num_workers=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
    
    def load_batch(self, indices):
        return [self.dataset[i] for i in indices]
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            # 비동기로 로드
            future = self.executor.submit(self.load_batch, batch_indices)
            batch = future.result()
            
            yield batch

# 4. 데이터 압축
import pickle
import gzip

def save_compressed(data, filename):
    """압축하여 저장"""
    with gzip.open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_compressed(filename):
    """압축 파일 로드"""
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)

print("=== 백엔드 최적화 기법 ===")
print("\n1. Prefetching")
print("   - 다음 배치를 미리 GPU로 전송")
print("   - 데이터 전송과 연산 오버랩")

print("\n2. 캐싱")
print("   - 자주 사용하는 데이터 메모리 저장")
print("   - LRU 정책")

print("\n3. 비동기 로딩")
print("   - 백그라운드에서 데이터 준비")
print("   - ThreadPoolExecutor 활용")

print("\n4. 압축")
print("   - 디스크 공간 절약")
print("   - 네트워크 전송 효율")
```

---
# 핵심 요약
---

## 이번 단원에서 배운 내용

### 1. 대용량 데이터 전략
```python
문제: 메모리에 올릴 수 없는 데이터

해결:
1. 청킹 (HDF5)
2. 메모리 맵
3. 스트리밍
4. 분산 처리
```

### 2. HDF5 vs Memmap
**HDF5**:
- 압축 지원
- 메타데이터
- 복잡한 구조
- 범용적

**Memmap**:
- 매우 빠름
- 간단함
- 원시 데이터
- 큰 디스크 필요

### 3. 스트리밍 데이터
- **IterableDataset**: 순차 접근
- **무한 스트림**: 실시간 데이터
- **Worker 분배**: 병렬 처리

### 4. 분산 학습
**DataParallel**:
- 간단
- 단일 머신
- GPU 0 병목

**DistributedDataParallel**:
- 효율적 (권장)
- 멀티 머신
- 복잡함

### 5. 백엔드 최적화
```python
1. Prefetching: 다음 배치 미리 로드
2. 캐싱: LRU 캐시
3. 비동기 로딩: ThreadPoolExecutor
4. 압축: gzip, lz4
5. Pin memory: CPU → GPU 전송 가속
6. Non-blocking: 전송과 연산 오버랩
```

### 실전 체크리스트
```python
데이터 크기에 따른 전략:

< 10GB:
  → 메모리에 로드

10GB ~ 100GB:
  → HDF5 또는 Memmap
  → num_workers=4~8

100GB ~ 1TB:
  → HDF5 + 압축
  → 스트리밍
  → 분산 학습

> 1TB:
  → 분산 파일 시스템 (HDFS)
  → 분산 학습 필수
  → 데이터 샘플링 고려
```

### DataLoader 최적화
```python
DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,        # CPU 코어의 절반
    pin_memory=True,      # GPU 전송 가속
    persistent_workers=True,  # Worker 재사용
    prefetch_factor=2     # 미리 로드할 배치 수
)
```

### 성능 비교
| 방법 | 속도 | 메모리 | 디스크 | 복잡도 |
|------|------|--------|--------|--------|
| 전체 로드 | 빠름 | 높음 | 낮음 | 낮음 |
| HDF5 | 중간 | 낮음 | 중간 | 중간 |
| Memmap | 빠름 | 낮음 | 높음 | 낮음 |
| 스트리밍 | 느림 | 매우 낮음 | 낮음 | 중간 |

### 백엔드 관점 정리
```python
딥러닝 데이터 처리 = 백엔드 시스템

- 청킹 = 페이지네이션
- 스트리밍 = 이벤트 스트림
- 캐싱 = Redis/Memcached
- 분산 = 마이크로서비스
- Prefetch = 프리로딩
```

이것으로 PyTorch 빅데이터 처리를 마칩니다! 🎉
