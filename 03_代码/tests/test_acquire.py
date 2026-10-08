from pathlib import Path
import hashlib
import json

import httpx
import pytest

from green_debt.acquire import (
    AcquisitionRunner,
    DownloadSpec,
    manifest_path_for,
)


PAYLOAD = b"country,year,value\nAAA,2000,1\n"


def make_spec(tmp_path: Path, **overrides: object) -> DownloadSpec:
    values: dict[str, object] = {
        "source_id": "fixture",
        "source_version": "v1",
        "url": "https://fixture.example/sample.csv",
        "allowed_hosts": ("fixture.example",),
        "destination": tmp_path / "raw" / "sample.csv",
        "expected_max_bytes": 1024,
        "expected_sha256": hashlib.sha256(PAYLOAD).hexdigest(),
        "projected_working_bytes": 2048,
    }
    values.update(overrides)
    return DownloadSpec(**values)


def test_acquire_writes_verified_file_manifest_and_audit_log(
    tmp_path: Path,
) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={
                "Content-Length": str(len(PAYLOAD)),
                "ETag": '"fixture-v1"',
            },
            content=PAYLOAD,
            request=request,
        )
    )
    runner = AcquisitionRunner.for_test(tmp_path, transport)
    spec = make_spec(tmp_path)

    record = runner.acquire(spec)

    assert spec.destination.read_bytes() == PAYLOAD
    assert not spec.destination.with_name("sample.csv.partial").exists()
    assert record.status == "downloaded"
    assert record.proxy_mode == "direct_only"
    assert record.bytes == len(PAYLOAD)
    assert record.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    manifest = json.loads(manifest_path_for(spec.destination).read_text())
    assert manifest["sha256"] == record.sha256
    log_rows = [
        json.loads(line)
        for line in (tmp_path / "下载日志.jsonl").read_text().splitlines()
    ]
    assert log_rows[-1]["status"] == "downloaded"
    assert "proxy" not in log_rows[-1]


def test_acquire_rejects_unlisted_host_before_http(tmp_path: Path) -> None:
    contacted = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal contacted
        contacted = True
        return httpx.Response(200, content=PAYLOAD, request=request)

    runner = AcquisitionRunner.for_test(
        tmp_path,
        httpx.MockTransport(handler),
    )
    spec = make_spec(
        tmp_path,
        url="https://unlisted.example/file.csv",
    )

    with pytest.raises(ValueError, match="host is not allowlisted"):
        runner.acquire(spec)

    assert contacted is False


def test_content_length_mismatch_removes_partial_file(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"Content-Length": "4"},
            content=b"12345",
            request=request,
        )
    )
    runner = AcquisitionRunner.for_test(tmp_path, transport)
    spec = make_spec(
        tmp_path,
        source_id="bad",
        url="https://fixture.example/bad.bin",
        destination=tmp_path / "bad.bin",
        expected_max_bytes=10,
        expected_sha256=None,
        projected_working_bytes=20,
    )

    with pytest.raises(RuntimeError, match="content length"):
        runner.acquire(spec)

    assert not (tmp_path / "bad.bin.partial").exists()
    assert not spec.destination.exists()


def test_redirect_is_not_followed_and_writes_no_body(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            302,
            headers={"Location": "https://cdn.example/file.zip"},
            content=b"redirect page must not be written",
            request=request,
        )

    runner = AcquisitionRunner.for_test(
        tmp_path,
        httpx.MockTransport(handler),
    )
    spec = make_spec(tmp_path, expected_sha256=None)

    with pytest.raises(RuntimeError, match="redirect refused"):
        runner.acquire(spec)

    assert len(requests) == 1
    assert not spec.destination.exists()
    assert not spec.destination.with_name("sample.csv.partial").exists()


def test_second_run_verifies_and_skips_existing_download(tmp_path: Path) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            headers={"Content-Length": str(len(PAYLOAD))},
            content=PAYLOAD,
            request=request,
        )

    runner = AcquisitionRunner.for_test(
        tmp_path,
        httpx.MockTransport(handler),
    )
    spec = make_spec(tmp_path)

    first = runner.acquire(spec)
    second = runner.acquire(spec)

    assert first.status == "downloaded"
    assert second.status == "verified_existing"
    assert request_count == 1


def test_response_larger_than_declared_cap_is_removed(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=b"123456",
            request=request,
        )
    )
    runner = AcquisitionRunner.for_test(tmp_path, transport)
    spec = make_spec(
        tmp_path,
        expected_max_bytes=5,
        expected_sha256=None,
    )

    with pytest.raises(RuntimeError, match="exceeds expected maximum"):
        runner.acquire(spec)

    assert not spec.destination.exists()
    assert not spec.destination.with_name("sample.csv.partial").exists()
