import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


class DataLakeManager:
    """A simple local data lake manager for raw, processed, and embedding artifacts."""

    def __init__(
        self,
        root_dir: Union[str, Path] = Path("data_storage"),
        raw_dir: str = "raw",
        processed_dir: str = "processed",
        embeddings_dir: str = "embeddings",
    ):
        self.root_dir = Path(root_dir)
        self.raw_dir = self.root_dir / raw_dir
        self.processed_dir = self.root_dir / processed_dir
        self.embeddings_dir = self.root_dir / embeddings_dir
        self._ensure_directories()

    def _ensure_directories(self):
        for directory in [self.raw_dir, self.processed_dir, self.embeddings_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _write_metadata(self, path: Path, metadata: Dict[str, Union[str, int, float, bool]]):
        with path.open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)

    def _current_version(self) -> str:
        return datetime.utcnow().strftime("%Y%m%d%H%M%S")

    def upload_raw(
        self,
        source_path: Union[str, Path],
        metadata: Optional[Dict[str, Union[str, int, float, bool]]] = None,
        dest_name: Optional[str] = None,
    ) -> str:
        """Upload an original raw file and store associated metadata."""
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")

        version = self._current_version()
        version_dir = self.raw_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        dest_name = dest_name or source_path.name
        dest_path = version_dir / dest_name
        shutil.copy2(source_path, dest_path)

        metadata = metadata or {}
        metadata.update(
            {
                "source_path": str(source_path),
                "uploaded_at": datetime.utcnow().isoformat() + "Z",
                "version": version,
                "filename": dest_name,
                "layer": "raw",
            }
        )
        self._write_metadata(version_dir / f"{dest_name}.metadata.json", metadata)

        return str(dest_path)

    def upload_processed(
        self,
        dataframe: pd.DataFrame,
        name: str,
        metadata: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    ) -> str:
        """Save a cleaned or transformed DataFrame as Parquet with metadata."""
        version = self._current_version()
        version_dir = self.processed_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = version_dir / f"{name}.parquet"
        dataframe.to_parquet(parquet_path, index=False)

        metadata = metadata or {}
        metadata.update(
            {
                "name": name,
                "saved_at": datetime.utcnow().isoformat() + "Z",
                "version": version,
                "layer": "processed",
                "rows": len(dataframe),
                "columns": dataframe.shape[1],
            }
        )
        self._write_metadata(version_dir / f"{name}.metadata.json", metadata)

        return str(parquet_path)

    def upload_embeddings(
        self,
        source_path: Union[str, Path],
        name: str,
        metadata: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    ) -> str:
        """Store a versioned embedding artifact such as a Word2Vec model file."""
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")

        version = self._current_version()
        version_dir = self.embeddings_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        dest_path = version_dir / f"{name}{source_path.suffix}"
        shutil.copy2(source_path, dest_path)

        metadata = metadata or {}
        metadata.update(
            {
                "name": name,
                "saved_at": datetime.utcnow().isoformat() + "Z",
                "version": version,
                "layer": "embeddings",
                "filename": dest_path.name,
            }
        )
        self._write_metadata(version_dir / f"{name}.metadata.json", metadata)

        return str(dest_path)

    def fetch_for_training(
        self,
        layer: str = "processed",
        version: Optional[str] = None,
        pattern: str = "*.parquet",
    ) -> List[pd.DataFrame]:
        """Fetch data artifacts for training from the requested layer."""
        base_dir = {
            "raw": self.raw_dir,
            "processed": self.processed_dir,
            "embeddings": self.embeddings_dir,
        }.get(layer)

        if base_dir is None:
            raise ValueError(f"Unknown layer: {layer}")

        if version:
            search_dirs = [base_dir / version]
        else:
            search_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

        dataframes = []
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for path in search_dir.glob(pattern):
                if path.suffix == ".parquet":
                    dataframes.append(pd.read_parquet(path))
        return dataframes

    def list_versions(self, layer: Optional[str] = None) -> Dict[str, List[str]]:
        """List all saved versions for the requested storage layers."""
        layers = [layer] if layer else ["raw", "processed", "embeddings"]
        result = {}
        for layer_name in layers:
            base_dir = {
                "raw": self.raw_dir,
                "processed": self.processed_dir,
                "embeddings": self.embeddings_dir,
            }.get(layer_name)
            if base_dir is None:
                raise ValueError(f"Unknown layer: {layer_name}")
            result[layer_name] = [d.name for d in sorted(base_dir.iterdir()) if d.is_dir()]
        return result


if __name__ == "__main__":
    manager = DataLakeManager(root_dir=Path("data_storage"))
    print("DataLakeManager initialized")
    print(manager.list_versions())
