#!/usr/bin/env python3
# =============================================================================
# main.py
# -----------------------------------------------------------------------------
# BUG FIX: The original main.py imported 'ChatAnalysisPipeline' which does
# not exist in pipeline.py. The correct import is run_pipeline().
# =============================================================================

import os
import sys
import argparse
import textwrap
from datetime import datetime

from config       import Paths, FlaskConfig
from src.pipeline import run_pipeline, get_pipeline_status


# --------------------------------------------------------------------------- #
# TERMINAL COLOURS                                                            #
# --------------------------------------------------------------------------- #

_USE_COLOUR = sys.stdout.isatty() and os.name != 'nt'

class C:
    RESET  = '\033[0m'  if _USE_COLOUR else ''
    BOLD   = '\033[1m'  if _USE_COLOUR else ''
    GREEN  = '\033[92m' if _USE_COLOUR else ''
    BLUE   = '\033[94m' if _USE_COLOUR else ''
    YELLOW = '\033[93m' if _USE_COLOUR else ''
    RED    = '\033[91m' if _USE_COLOUR else ''
    CYAN   = '\033[96m' if _USE_COLOUR else ''
    DIM    = '\033[2m'  if _USE_COLOUR else ''


def _line(char='─', width=54):
    print(C.DIM + char * width + C.RESET)

def _header(title: str):
    print()
    print(C.BOLD + C.BLUE + f'── {title} ' + '─' * max(0, 48 - len(title)) + C.RESET)

def _row(label: str, value, width=26, colour=''):
    val_str = str(value) if value is not None else '–'
    print(f'  {C.DIM}{label:<{width}}{C.RESET}{colour}{val_str}{C.RESET}')

def _success(msg: str):
    print(f'{C.GREEN}  ✓  {msg}{C.RESET}')

def _warn(msg: str):
    print(f'{C.YELLOW}  ⚠  {msg}{C.RESET}')

def _error(msg: str):
    print(f'{C.RED}  ✗  {msg}{C.RESET}')

def _tick(label: str, ok: bool):
    icon = (C.GREEN + '✓') if ok else (C.RED + '✗')
    print(f'  {icon}{C.RESET}  {label}')


def print_banner():
    width = 56
    print()
    print(C.BOLD + C.BLUE + '═' * width + C.RESET)
    print(C.BOLD + C.BLUE + '  WhatsApp NLP Analysis System  v1.0.0' + C.RESET)
    print(C.BOLD + C.BLUE + '  Chaitanya  ·  Anudeep  ·  Ram Teja' + C.RESET)
    print(C.BOLD + C.BLUE + '═' * width + C.RESET)


def print_results(output):
    print_banner()

    _header('Chat Summary')
    cs = output.chat_summary
    _row('Total Messages',    cs.get('total_messages',    '–'))
    _row('Text Messages',     cs.get('total_text_msgs',   '–'))
    _row('Media Messages',    cs.get('total_media_msgs',  '–'))
    _row('Participants',      cs.get('total_senders',     '–'))
    _row('Date Range',
         f"{cs.get('date_start','?')} → {cs.get('date_end','?')}")
    _row('Days Spanned',      cs.get('total_days',        '–'))
    _row('Most Active User',  cs.get('most_active_user',  '–'), colour=C.CYAN)
    _row('Total Words',       cs.get('total_words',       '–'))
    _row('Avg Words/Message', cs.get('avg_words_per_msg', '–'))

    _header('Sentiment Analysis')
    mood       = output.overall_mood
    mood_label = mood.get('overall_label', '–')
    mood_color = C.GREEN if mood_label == 'Positive' else C.RED if mood_label == 'Negative' else C.YELLOW
    _row('Overall Mood',         mood.get('mood_description',  '–'), colour=mood_color)
    _row('Mood Score',           f"{mood.get('overall_score', 0):+.3f}", colour=mood_color)
    _row('Positive Messages',    f"{mood.get('positive_pct', 0):.1f}%",  colour=C.GREEN)
    _row('Neutral Messages',     f"{mood.get('neutral_pct',  0):.1f}%",  colour=C.YELLOW)
    _row('Negative Messages',    f"{mood.get('negative_pct', 0):.1f}%",  colour=C.RED)

    if not output.sentiment_by_sender.empty:
        _header('Sentiment by Sender')
        print(f"  {'Sender':<20} {'Msgs':>6}  {'Pos%':>6}  {'Neu%':>6}  {'Neg%':>6}  {'Avg':>7}")
        _line('·')
        for _, row in output.sentiment_by_sender.iterrows():
            pos = row.get('positive_pct', 0)
            neg = row.get('negative_pct', 0)
            avg = row.get('avg_score',    0)
            sc  = C.GREEN if avg > 0.05 else C.RED if avg < -0.05 else C.YELLOW
            print(
                f"  {str(row['sender']):<20}"
                f" {int(row.get('total', 0)):>6}"
                f"  {C.GREEN}{pos:>5.1f}%{C.RESET}"
                f"  {C.YELLOW}{row.get('neutral_pct', 0):>5.1f}%{C.RESET}"
                f"  {C.RED}{neg:>5.1f}%{C.RESET}"
                f"  {sc}{avg:>+6.3f}{C.RESET}"
            )

    _header('NLP Summary')
    nlp = output.nlp_summary
    _row('Total Words',        nlp.get('total_words',      '–'))
    _row('Unique Words',       nlp.get('unique_words',     '–'))
    _row('Peak Hour',          nlp.get('peak_hour',        '–'), colour=C.CYAN)
    _row('Most Active Day',    nlp.get('most_active_day',  '–'), colour=C.CYAN)
    _row('Top Word',           nlp.get('top_word',         '–'), colour=C.BOLD)
    _row('Fastest Responder',  nlp.get('fastest_responder','–'), colour=C.GREEN)

    _header('ML Model')
    meta = output.model_metadata
    if meta:
        acc       = meta.get('accuracy', 0)
        acc_color = C.GREEN if acc >= 0.80 else C.YELLOW if acc >= 0.65 else C.RED
        _row('Best Model',        meta.get('model_name',       '–'), colour=C.CYAN)
        _row('Test Accuracy',     f"{acc * 100:.1f}%",               colour=acc_color)
        _row('F1 Score',          f"{meta.get('f1_score', 0) * 100:.1f}%")
        _row('Cross-Val Mean',    f"{meta.get('cross_val_mean',  0) * 100:.1f}%")
        _row('Training Samples',  meta.get('training_samples', '–'))
        _row('Trained At',        meta.get('trained_at',       '–'))
        _row('ML This Run',       'Yes — trained fresh' if output.ml_trained else 'Loaded saved model')
    else:
        _warn('No ML model trained — chat may have < 50 messages or < 20 usable training samples.')

    _header('Pipeline Info')
    _row('Mode',             output.mode_used)
    _row('Processing Time',  f'{output.processing_time}s')
    _row('Run Timestamp',    output.run_timestamp)
    _row('Charts Location',  Paths.CHARTS_DIR)

    print()
    _line('═')
    _success(f'Analysis complete in {output.processing_time}s')
    print(f'  {C.DIM}Charts saved to : {Paths.CHARTS_DIR}{C.RESET}')
    print(f'  {C.DIM}Start web UI    : python main.py --serve{C.RESET}')
    print()


def print_status():
    print_banner()
    _header('System Status')
    s = get_pipeline_status()
    _tick('ML model trained',        s.get('model_trained',    False))
    _tick('Output directory exists', s.get('output_dir_exists',False))
    charts = s.get('charts_available', 0)
    _tick(f'Charts available ({charts}/10)', charts == 10)
    meta = s.get('model_metadata', {})
    if meta:
        _header('Last Model Performance')
        _row('Accuracy',   f"{meta.get('accuracy', 0) * 100:.1f}%")
        _row('F1 Score',   f"{meta.get('f1_score', 0) * 100:.1f}%")
        _row('Model Name', meta.get('model_name', '–'))
        _row('Trained At', meta.get('trained_at', '–'))
    else:
        _warn('No trained model found. Run analysis first.')
    print()


def save_report(output, file_path: str) -> str:
    os.makedirs(Paths.RESULTS_DIR, exist_ok=True)
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(Paths.RESULTS_DIR, f'report_{timestamp}.txt')
    lines = [
        '=' * 60, 'WHATSAPP NLP ANALYSIS REPORT',
        f'File      : {file_path}', f'Generated : {output.run_timestamp}',
        f'Mode      : {output.mode_used}', '=' * 60, '',
        '── CHAT SUMMARY ──────────────────────────────────────',
    ]
    for k, v in output.chat_summary.items():
        lines.append(f'  {k:<25}: {v}')
    lines += ['', '── OVERALL SENTIMENT ─────────────────────────────────']
    for k, v in output.overall_mood.items():
        lines.append(f'  {k:<25}: {v}')
    if not output.sentiment_by_sender.empty:
        lines += ['', '── SENTIMENT BY SENDER ───────────────────────────────']
        lines.append(output.sentiment_by_sender.to_string(index=False))
    lines += ['', '── NLP SUMMARY ───────────────────────────────────────']
    for k, v in output.nlp_summary.items():
        lines.append(f'  {k:<25}: {v}')
    lines += ['', '── ML MODEL ──────────────────────────────────────────']
    if output.model_metadata:
        for k, v in output.model_metadata.items():
            lines.append(f'  {k:<25}: {v}')
    else:
        lines.append('  No model metadata available.')
    lines += ['', '── CHARTS ────────────────────────────────────────────']
    for key, path in output.chart_paths.items():
        status = 'OK' if path and os.path.exists(path) else 'FAILED'
        lines.append(f'  [{status}] {key:<30} {os.path.basename(path) if path else ""}')
    lines += ['', '=' * 60, f'Processing time : {output.processing_time}s', '=' * 60]
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return report_path


def main():
    parser = argparse.ArgumentParser(
        prog='main.py',
        description=textwrap.dedent('''
            WhatsApp NLP Analysis System — CLI Runner
            -----------------------------------------
            Examples:
              python main.py chat.txt
              python main.py chat.txt --retrain --report
              python main.py --serve
              python main.py --status
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('file',        nargs='?',        default=None)
    parser.add_argument('--retrain',   action='store_true', default=False)
    parser.add_argument('--report',    action='store_true', default=False)
    parser.add_argument('--serve',     action='store_true', default=False)
    parser.add_argument('--status',    action='store_true', default=False)
    parser.add_argument('--port',      type=int, default=FlaskConfig.PORT)
    args = parser.parse_args()

    Paths.create_all_dirs()

    if args.status:
        print_status()
        sys.exit(0)

    if args.serve:
        print_banner()
        print()
        _success(f'Starting Flask server → http://localhost:{args.port}')
        print(f'  {C.DIM}Press Ctrl+C to stop{C.RESET}')
        print()
        from app import app
        app.run(host=FlaskConfig.HOST, port=args.port, debug=FlaskConfig.DEBUG)
        sys.exit(0)

    if not args.file:
        parser.print_help()
        print()
        _error('No chat file provided.')
        print(f'  {C.DIM}Example: python main.py chat.txt{C.RESET}')
        sys.exit(1)

    if not os.path.exists(args.file):
        _error(f'File not found: {args.file}')
        sys.exit(1)

    if not args.file.lower().endswith('.txt'):
        _error('File must be a .txt WhatsApp export.')
        sys.exit(1)

    print_banner()
    print()
    print(f'  {C.DIM}File    : {args.file}{C.RESET}')
    print(f'  {C.DIM}Retrain : {args.retrain}{C.RESET}')
    print()

    output = run_pipeline(args.file, force_retrain=args.retrain)
    if not output.success:
        print()
        _error(f'Pipeline failed: {output.error_message}')
        sys.exit(1)

    print_results(output)

    if args.report:
        report_path = save_report(output, args.file)
        _success(f'Report saved → {report_path}')
        print()


if __name__ == '__main__':
    main()