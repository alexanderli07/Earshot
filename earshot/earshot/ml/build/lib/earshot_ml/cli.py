"""Command-line interface for Earshot's offline sound detection pipeline."""

import argparse
import json
import math
import time

from . import config
from .artifacts import ArtifactError, download_artifact
from .core import EarshotML, TeachStore, TeachStoreError
from .pipeline import (AudioDeviceError, AudioFileError,
                       InterpreterBackendError, MicStream,
                       ModelContractError, YamNet, record)


class CLIError(RuntimeError):
    """An expected command-line usage error that does not need a traceback."""


def cmd_download(args):
    """Install both pinned artifacts and validate their shared contract."""
    for artifact in (config.MODEL_ARTIFACT, config.CLASS_MAP_ARTIFACT):
        downloaded = download_artifact(artifact)
        status = "downloaded" if downloaded else "cached"
        print(f"{status} {artifact.path.name}")

    YamNet(config.MODEL_PATH, config.CLASS_MAP_PATH)
    print("validated model and class map")
    print("done")


def cmd_top5(args):
    yamnet = YamNet(config.MODEL_PATH, config.CLASS_MAP_PATH)
    print("listening... Ctrl-C to stop")
    for waveform in MicStream(device=args.device).windows():
        scores, _ = yamnet.infer(waveform)
        top = "  |  ".join(
            f"{name} {score:.2f}" for name, score in yamnet.top(scores, k=5)
        )
        peak = float(abs(waveform).max())
        print(f"\rpeak {peak:.2f}  {top:<120}", end="", flush=True)


def cmd_run(args):
    def on_event(event):
        stamp = time.strftime("%H:%M:%S", time.localtime(event["timestamp"]))
        print(f"\n[{stamp}] EVENT {json.dumps(event)}")

    engine = EarshotML(
        on_event=on_event,
        device=args.device,
        model_path=config.MODEL_PATH,
        class_map_path=config.CLASS_MAP_PATH,
        taught_store_path=config.TAUGHT_STORE_PATH,
    )
    learned = engine.learned_sounds()
    if learned:
        print(f"taught sounds loaded: {[sound['name'] for sound in learned]}")
    print("listening for events... Ctrl-C to stop")
    engine.run()


def _validated_teach_name(name):
    trimmed = name.strip()
    if not trimmed:
        raise CLIError("teach name must be a non-empty string")
    reserved_names = {
        entry["label"].strip().casefold() for entry in config.EVENT_MAP
    }
    if trimmed.casefold() in reserved_names:
        raise CLIError(
            f"teach name {trimmed!r} conflicts with a pretrained label"
        )
    return trimmed


def cmd_teach(args):
    name = _validated_teach_name(args.name)
    if args.record < 0:
        raise CLIError("--record must be a non-negative integer")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise CLIError("--seconds must be positive and finite")
    clips = list(args.clips)
    if not clips and args.record <= 0:
        raise CLIError("give wav files or --record N")

    engine = EarshotML(
        device=args.device,
        model_path=config.MODEL_PATH,
        class_map_path=config.CLASS_MAP_PATH,
        taught_store_path=config.TAUGHT_STORE_PATH,
    )
    for index in range(args.record):
        input(
            "press Enter, then make the sound "
            f"(clip {index + 1}/{args.record}, {args.seconds:.0f}s)... "
        )
        time.sleep(0.2)
        clips.append(record(args.seconds, device=args.device))
        print("  captured")

    stored = engine.teach(name, clips)
    print(
        f"taught {name!r} from {stored} clips; "
        f"known sounds: {[sound['name'] for sound in engine.learned_sounds()]}"
    )


def _teach_store():
    return TeachStore(
        path=config.TAUGHT_STORE_PATH,
        cutoff=config.TAUGHT_SIMILARITY_CUTOFF,
    )


def cmd_sounds(args):
    for sound in _teach_store().learned():
        print(f"  {sound['name']}  ({sound['clips']} clips)")


def cmd_forget(args):
    store = _teach_store()
    removed = store.forget(args.name)
    store.save()
    print(f"removed {removed} clips of {args.name!r}")


def _build_parser():
    parser = argparse.ArgumentParser(
        description="One CLI for Earshot's offline sound detection pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download").set_defaults(fn=cmd_download)

    top5 = subparsers.add_parser("top5")
    top5.add_argument(
        "--device", type=int, default=None, help="sounddevice input device index"
    )
    top5.set_defaults(fn=cmd_top5)

    run = subparsers.add_parser("run")
    run.add_argument("--device", type=int, default=None)
    run.set_defaults(fn=cmd_run)

    teach = subparsers.add_parser("teach")
    teach.add_argument("name", help="name of the sound")
    teach.add_argument("clips", nargs="*", help="wav files (16 kHz-ish, mono)")
    teach.add_argument(
        "--record",
        type=int,
        default=0,
        metavar="N",
        help="record N clips from the mic instead",
    )
    teach.add_argument(
        "--seconds", type=float, default=2.0, help="length of each recorded clip"
    )
    teach.add_argument("--device", type=int, default=None)
    teach.set_defaults(fn=cmd_teach)

    subparsers.add_parser("sounds").set_defaults(fn=cmd_sounds)

    forget = subparsers.add_parser("forget")
    forget.add_argument("name")
    forget.set_defaults(fn=cmd_forget)

    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except ArtifactError as exc:
        parser.exit(
            1,
            f"error: {exc}\n"
            "Retry `earshot download`; check the artifact source or "
            "connection if the error continues.\n",
        )
    except (
        CLIError,
        AudioDeviceError,
        AudioFileError,
        InterpreterBackendError,
        ModelContractError,
        TeachStoreError,
        FileNotFoundError,
    ) as exc:
        parser.exit(1, f"error: {exc}\n")
    except KeyboardInterrupt:
        print()
        return None


if __name__ == "__main__":
    main()
