"""Command-line interface for Earshot's offline sound detection pipeline."""

import argparse
import json
import math
import time
from pathlib import Path

from . import config
from .alarm_data import AlarmDataError, collect_files, collect_recordings
from .alarm_model import AlarmModelError, load_optional_alarm_head
from .artifacts import ArtifactError, download_artifact
from .core import EarshotML, TeachStore, TeachStoreError
from .pipeline import (AudioDeviceError, AudioFileError,
                       InterpreterBackendError, MicStream,
                       ModelContractError, YamNet, record)


class CLIError(RuntimeError):
    """An expected command-line usage error that does not need a traceback."""


def _train_alarm(*args, **kwargs):
    from .alarm_training import train_alarm

    return train_alarm(*args, **kwargs)


def _evaluate_alarm(*args, **kwargs):
    from .alarm_training import evaluate_alarm

    return evaluate_alarm(*args, **kwargs)


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
    alarm_head = load_optional_alarm_head(
        config.ALARM_MODEL_PATH,
        yamnet_model_path=config.MODEL_PATH,
        class_map_path=config.CLASS_MAP_PATH,
    )
    print("listening... Ctrl-C to stop")
    for waveform in MicStream(device=args.device).windows():
        scores, embedding = yamnet.infer(waveform)
        top = "  |  ".join(
            f"{name} {score:.2f}" for name, score in yamnet.top(scores, k=5)
        )
        if alarm_head is not None:
            top += (
                "  |  fire_smoke_alarm "
                f"{alarm_head.score(embedding):.2f}"
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
        alarm_model_path=config.ALARM_MODEL_PATH,
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
    normalized = trimmed.casefold()
    if normalized in config.RESERVED_EVENT_LABELS:
        label_kind = (
            "trained alarm"
            if normalized == config.ALARM_EVENT_LABEL.strip().casefold()
            else "pretrained"
        )
        raise CLIError(
            f"teach name {trimmed!r} conflicts with a {label_kind} label"
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


def _before_alarm_capture(index, count, seconds):
    input(
        "press Enter, then make the sound "
        f"(clip {index}/{count}, {seconds:g}s)... "
    )
    time.sleep(0.2)


def cmd_collect(args):
    if args.record < 0:
        raise CLIError("--record must be a non-negative integer")
    minimum_seconds = config.WINDOW_SAMPLES / config.SAMPLE_RATE
    if (
        not math.isfinite(args.seconds)
        or args.seconds < minimum_seconds
    ):
        raise CLIError(
            "--seconds must record at least one model window "
            f"({minimum_seconds:g})"
        )
    if not args.wavs and args.record <= 0:
        raise CLIError("give wav files or --record N")

    stored = []
    if args.wavs:
        stored.extend(
            collect_files(
                args.label,
                args.wavs,
                args.data_dir,
                source_group=args.source_group,
            )
        )
    if args.record:
        stored.extend(
            collect_recordings(
                args.label,
                args.record,
                args.seconds,
                args.data_dir,
                source_group=args.source_group,
                device=args.device,
                recorder=record,
                before_capture=_before_alarm_capture,
            )
        )
    for path in stored:
        print(f"  {path}")
    print(f"stored {len(stored)} clips")


def cmd_train_alarm(args):
    output = Path(args.output)
    report_path = (
        Path(config.ALARM_REPORT_PATH)
        if output == Path(config.ALARM_MODEL_PATH)
        else output.with_name("fire_smoke_alarm_report.json")
    )
    report = _train_alarm(
        args.data_dir,
        output,
        report_path,
        seed=args.seed,
    )
    metrics = report.oof_metrics
    print(
        f"trained fire_smoke_alarm threshold "
        f"{report.deployment_threshold:.3f}; "
        f"recall {metrics.positive_groups_triggered}/"
        f"{metrics.positive_groups_total}; "
        f"negative groups {metrics.negative_groups_triggered}/"
        f"{metrics.negative_groups_total}; "
        f"false triggers/min {metrics.false_triggers_per_minute:.3f}"
    )


def cmd_evaluate_alarm(args):
    report = _evaluate_alarm(args.data_dir, args.model)
    for item in report.metrics.files:
        print(f"  {item['label']:<9} {item['triggered']!s:<5} {item['path']}")
    print(json.dumps(report.payload, sort_keys=True))


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

    collect = subparsers.add_parser("collect")
    collect.add_argument("label", choices=("alarm", "not_alarm"))
    collect.add_argument("wavs", nargs="*", type=Path, metavar="WAV")
    collect.add_argument("--record", type=int, default=0, metavar="N")
    collect.add_argument("--seconds", type=float, default=5.0, metavar="S")
    collect.add_argument("--device", type=int, default=None, metavar="INDEX")
    collect.add_argument(
        "--data-dir", type=Path, default=config.ALARM_DATA_DIR, metavar="PATH"
    )
    collect.add_argument("--source-group", default=None, metavar="NAME")
    collect.set_defaults(fn=cmd_collect)

    train_alarm = subparsers.add_parser("train-alarm")
    train_alarm.add_argument(
        "--data-dir", type=Path, default=config.ALARM_DATA_DIR, metavar="PATH"
    )
    train_alarm.add_argument(
        "--output", type=Path, default=config.ALARM_MODEL_PATH, metavar="PATH"
    )
    train_alarm.add_argument("--seed", type=int, default=0)
    train_alarm.set_defaults(fn=cmd_train_alarm)

    evaluate_alarm = subparsers.add_parser("evaluate-alarm")
    evaluate_alarm.add_argument(
        "--data-dir", type=Path, default=config.ALARM_DATA_DIR, metavar="PATH"
    )
    evaluate_alarm.add_argument(
        "--model", type=Path, default=config.ALARM_MODEL_PATH, metavar="PATH"
    )
    evaluate_alarm.set_defaults(fn=cmd_evaluate_alarm)

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
        AlarmDataError,
        AlarmModelError,
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
