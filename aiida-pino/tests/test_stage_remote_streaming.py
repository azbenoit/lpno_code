"""Staging uploads member by member; the local transient is one member, not the dataset.

The bulk path built the whole cache and uploaded it after: two concurrent ~40 GB stagings
took root from 100 GB to 19 GB in three minutes (2026-08-27), and the operator had checked
only the TARGET's disk because the local cost was invisible in the command's contract.
Nothing needed decompressing -- stored and staged sizes matched at 1.0x -- so the fix is to
interleave, not to compress.
"""

import hashlib
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from aiida_pino.utils import stage_remote

MEMBERS = 4
MEMBER_BYTES = 1000


class _Collection:
    """Duck-typed collection: N members, one payload file each, plus a manifest epilogue."""

    uuid = "cafecafe-0000-0000-0000-000000000000"
    pk = 1

    def __init__(self, peak_log):
        self._peak_log = peak_log

    def get_dataset_metadata(self):
        return {"derivation_profile": {}}

    def iter_member_simulations(self):
        return [object()] * MEMBERS

    def export_runtime_arraydir(self, target, *, runtime_fields=None,
                                allow_pack_index=True, on_member_staged=None,
                                allow_full_resolution=False):
        target = Path(target)
        assert allow_pack_index is False, "remote staging must never write a pack index"
        for index in range(MEMBERS):
            rel = f"members/simulation_{index:04d}"
            member = target / rel
            member.mkdir(parents=True)
            (member / "payload.npz").write_bytes(os.urandom(MEMBER_BYTES))
            (member / "manifest.json").write_text("{}")
            # The invariant under test: when the callback runs, earlier members are GONE.
            live = [p.name for p in (target / "members").iterdir() if p.is_dir()]
            self._peak_log.append(len(live))
            if on_member_staged is not None:
                on_member_staged(rel)
        (target / "manifest.json").write_text('{"members": %d}' % MEMBERS)
        return target


class _Transport:
    """Records every upload; answers the verification queries from those records."""

    def __init__(self):
        self.files: dict[str, tuple[int, str]] = {}
        self.remote_dir = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def path_exists(self, path):
        return False

    def rmtree(self, path):
        pass

    def makedirs(self, path, ignore_existing=False):
        pass

    def _record(self, local: Path, remote: str):
        data = local.read_bytes()
        self.files[remote] = (len(data), hashlib.md5(data).hexdigest())

    def puttree(self, local, remote):
        base = Path(local)
        for path in base.rglob("*"):
            if path.is_file():
                self._record(path, f"{remote}/{path.relative_to(base).as_posix()}")

    def put(self, local, remote):
        self._record(Path(local), remote)

    def exec_command_wait(self, command):
        prefix = f"{self.remote_dir}/"
        rows = {k[len(prefix):]: v for k, v in self.files.items() if k.startswith(prefix)}
        if "find" in command:
            return 0, "".join(f"{size} {rel}\0" for rel, (size, _) in rows.items()), ""
        return 0, "".join(f"{digest} {rel}\0" for rel, (_, digest) in rows.items()), ""


class _Computer:
    label = "fakebox"
    transport_type = "core.ssh_async"

    def __init__(self, transport):
        self._t = transport

    def get_transport(self):
        return self._t


class StreamingStagingTest(unittest.TestCase):
    def _run(self, tmp, verify=True):
        peaks: list[int] = []
        collection = _Collection(peaks)
        transport = _Transport()
        computer = _Computer(transport)
        registered = {}

        def register(comp, cache_name, remote_path, verified):
            registered.update(cache_name=cache_name, remote_path=remote_path, verified=verified)
            transport.remote_dir = remote_path
            return mock.Mock()

        with mock.patch("aiida_pino.utils.data_nodes.register_remote_runtime_cache",
                               side_effect=register), \
             mock.patch.object(stage_remote, "remote_cache_root",
                               side_effect=lambda t, o=None: (transport.__setattr__("remote_dir_root", "/remote"), "/remote")[1]), \
             mock.patch("aiida_pino.utils.runtime_cache.runtime_cache_root",
                        return_value=Path(tmp)):
            # remote_dir must be known to the fake transport before verification runs.
            original_verify = stage_remote.verify_remote_cache

            def verify_hook(t, remote_dir, expected, **kw):
                t.remote_dir = remote_dir
                return original_verify(t, remote_dir, expected, **kw)

            with mock.patch.object(stage_remote, "verify_remote_cache", side_effect=verify_hook):
                result = stage_remote.stage_collection_to_computer(
                    collection, computer, verify_content=verify
                )
        return result, peaks, transport, registered, Path(tmp)

    def test_peak_local_use_is_one_member(self):
        with TemporaryDirectory() as tmp:
            result, peaks, transport, registered, root = self._run(tmp)
        # At every callback exactly ONE member directory existed: the one just written.
        self.assertEqual(peaks, [1] * MEMBERS, peaks)

    def test_everything_arrives_and_is_verified(self):
        with TemporaryDirectory() as tmp:
            result, peaks, transport, registered, root = self._run(tmp)
        # 4 members x 2 files + the top-level manifest.
        self.assertEqual(len(transport.files), MEMBERS * 2 + 1)
        self.assertEqual(result["file_count"], MEMBERS * 2 + 1)
        self.assertEqual(registered["verified"]["file_count"], MEMBERS * 2 + 1)

    def test_no_local_copy_survives(self):
        with TemporaryDirectory() as tmp:
            result, peaks, transport, registered, root = self._run(tmp)
            leftovers = [p.name for p in root.iterdir() if p.is_dir()]
        self.assertEqual(leftovers, [], "streaming must leave no local cache behind")

    def test_a_corrupted_transfer_is_refused_and_retracted(self):
        with TemporaryDirectory() as tmp:
            peaks: list[int] = []
            transport = _Transport()
            # Corrupt one recorded file so checksum verification must fail.
            original = transport._record

            def corrupt(local, remote):
                original(local, remote)
                if remote.endswith("simulation_0002/payload.npz"):
                    size, _ = transport.files[remote]
                    transport.files[remote] = (size, "0" * 32)
            transport._record = corrupt
            computer = _Computer(transport)
            retracted = []
            with mock.patch("aiida_pino.utils.data_nodes.register_remote_runtime_cache") as reg, \
                 mock.patch.object(stage_remote, "retract_remote_runtime_cache",
                                   side_effect=lambda c, n: retracted.append(n)), \
                 mock.patch.object(stage_remote, "remote_cache_root",
                                   side_effect=lambda t, o=None: "/remote"), \
                 mock.patch("aiida_pino.utils.runtime_cache.runtime_cache_root",
                            return_value=Path(tmp)):
                original_verify = stage_remote.verify_remote_cache

                def verify_hook(t, remote_dir, expected, **kw):
                    t.remote_dir = remote_dir
                    return original_verify(t, remote_dir, expected, **kw)

                with mock.patch.object(stage_remote, "verify_remote_cache", side_effect=verify_hook):
                    with self.assertRaises(RuntimeError):
                        stage_remote.stage_collection_to_computer(
                            _Collection(peaks), computer, verify_content=True
                        )
                reg.assert_not_called()
            self.assertEqual(len(retracted), 1)

    def test_dry_run_builds_nothing(self):
        with TemporaryDirectory() as tmp:
            transport = _Transport()
            with mock.patch("aiida_pino.utils.runtime_cache.runtime_cache_root",
                            return_value=Path(tmp)):
                result = stage_remote.stage_collection_to_computer(
                    _Collection([]), _Computer(transport), dry_run=True
                )
            self.assertFalse(result["staged"])
            self.assertEqual(result["member_count"], MEMBERS)
            self.assertEqual(transport.files, {}, "dry run must transfer nothing")
            self.assertEqual([p.name for p in Path(tmp).iterdir()], [],
                             "dry run must build nothing")


if __name__ == "__main__":
    unittest.main()


class SourceSideProtectionTest(StreamingStagingTest):
    """The destination gate protected the destination; the SOURCE was unprotected.

    2026-09-09: staging a materialised variant to a compute machine took the submitting machine's root
    from 35 GB free to 22 GB -- below the FLOW-3D free-space floor -- and the `.stream-`
    directory survived the process being killed, sitting at 62 GB until a disk monitor fired.
    """

    def test_a_dead_builders_stream_dir_is_swept_before_building(self):
        import subprocess

        proc = subprocess.Popen(["true"])
        proc.wait()
        dead = proc.pid  # reaped: guaranteed gone
        with TemporaryDirectory() as tmp:
            orphan = Path(tmp) / f"deadbeef.stream-{dead}"
            (orphan / "members").mkdir(parents=True)
            (orphan / "members" / "junk").write_bytes(b"x" * 100)
            live = Path(tmp) / f"cafe.stream-{os.getppid()}"  # another builder, still alive
            live.mkdir()
            self._run(tmp)
            self.assertFalse(orphan.exists(), "orphaned build dir must be swept")
            self.assertTrue(live.exists(), "a live builder's dir is not ours to remove")

    def test_the_source_disk_is_gated_before_building(self):
        with TemporaryDirectory() as tmp, \
             mock.patch("aiida_pino.utils.runtime_cache.refuse_build_that_cannot_fit") as gate:
            self._run(tmp)
        gate.assert_called_once()
        self.assertEqual(Path(gate.call_args.args[0]), Path(tmp))
