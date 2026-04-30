from flask import Flask, render_template, request, jsonify, send_from_directory, abort
import re
import shlex
import subprocess
import threading
import uuid
from pathlib import Path

app = Flask(__name__)
ROOT = Path(__file__).resolve().parent
GENOME_DIR = ROOT / "genomes"
KMER_OUT_DIR = ROOT / "kmergenie_output"
COGRAM_DIR = ROOT / "cogram"

ORGANISMS = {
    "nasuia_deltocephalinicola": {
        "label": "Nasuia deltocephalinicola",
        "url": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/186/945/GCF_000186945.1_ASM18694v1/GCF_000186945.1_ASM18694v1_genomic.fna.gz",
    }
}

STATE = {
    "organism": None,
    "organism_label": None,
    "genome_file": None,
    "prefix": None,
    "r1": None,
    "r2": None,
    "combined": None,
    "best_k": None,
}

JOBS = {}
JOBS_LOCK = threading.Lock()


def run_bash_stream(command: str, cwd: Path, job_id: str):
    process = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in iter(process.stdout.readline, ""):
        with JOBS_LOCK:
            JOBS[job_id]["log"] += line

    process.stdout.close()
    code = process.wait()
    with JOBS_LOCK:
        JOBS[job_id]["done"] = True
        JOBS[job_id]["returncode"] = code


def start_job(command: str, cwd: Path, meta=None):
    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {
            "log": "",
            "done": False,
            "returncode": None,
            "meta": meta or {},
        }
    t = threading.Thread(target=run_bash_stream, args=(command, cwd, job_id), daemon=True)
    t.start()
    return job_id


def conda_cmd(env_name: str, inner: str):
    return (
        "source ~/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate {shlex.quote(env_name)} && "
        f"{inner}"
    )


def parse_kmergenie_dat(dat_path: Path):
    rows = []
    if not dat_path.exists():
        return None, rows

    with dat_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("k "):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                rows.append({"k": int(parts[0]), "genomic_kmers": int(parts[1])})

    best_k = max(rows, key=lambda x: x["genomic_kmers"])["k"] if rows else None
    return best_k, rows


def _extract_int(text, pattern):
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def _extract_float(text, pattern):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


@app.route("/")
def index():
    return render_template("index.html", organisms=ORGANISMS, step=0)


@app.route("/step/<int:step>")
def step_page(step):
    if step not in (1, 2, 3, 4):
        abort(404)
    return render_template("index.html", organisms=ORGANISMS, step=step)


@app.route("/api/restart", methods=["POST"])
def restart_pipeline():
    with JOBS_LOCK:
        JOBS.clear()
    STATE.update({
        "organism": None,
        "organism_label": None,
        "genome_file": None,
        "prefix": None,
        "r1": None,
        "r2": None,
        "combined": None,
        "best_k": None,
    })
    return jsonify({"ok": True})


@app.route("/api/setup_genome/start", methods=["POST"])
def setup_genome_start():
    data = request.get_json(force=True)
    org_key = data.get("organism")
    if org_key not in ORGANISMS:
        return jsonify({"ok": False, "error": "Invalid organism"}), 400

    info = ORGANISMS[org_key]
    prefix = org_key.replace(" ", "_").lower()

    GENOME_DIR.mkdir(parents=True, exist_ok=True)
    gz_name = Path(info["url"]).name
    gz_path = GENOME_DIR / gz_name
    fasta_path = GENOME_DIR / f"{prefix}.fasta"

    cmd = (
        f"wget -q -O {shlex.quote(str(gz_path))} {shlex.quote(info['url'])} && "
        f"gunzip -f {shlex.quote(str(gz_path))} && "
        f"mv -f {shlex.quote(str(gz_path.with_suffix('')))} {shlex.quote(str(fasta_path))}"
    )

    STATE.update({
        "organism": org_key,
        "organism_label": info["label"],
        "genome_file": str(fasta_path),
        "prefix": prefix,
        "r1": str(ROOT / f"{prefix}_sim_R1.fastq"),
        "r2": str(ROOT / f"{prefix}_sim_R2.fastq"),
        "combined": str(ROOT / f"{prefix}_combined.fastq"),
        "best_k": None,
    })

    job_id = start_job(cmd, ROOT, {"step": "genome"})
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/simulate_reads/start", methods=["POST"])
def simulate_reads_start():
    data = request.get_json(force=True)
    n_reads = int(data.get("n_reads", 3000000))
    model = data.get("model", "hiseq")

    genome_file = STATE.get("genome_file")
    prefix = STATE.get("prefix")
    if not genome_file or not prefix:
        return jsonify({"ok": False, "error": "Go to Step 1 first and prepare genome."}), 400

    cmd = conda_cmd(
        "sim_env",
        " ".join([
            "iss generate",
            f"--genomes {shlex.quote(genome_file)}",
            f"--model {shlex.quote(model)}",
            f"--output {shlex.quote(prefix + '_sim')}",
            f"--n_reads {n_reads}",
            "--abundance uniform",
        ]),
    )

    job_id = start_job(cmd, ROOT, {"step": "simulate"})
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/run_kmergenie/start", methods=["POST"])
def run_kmergenie_start():
    data = request.get_json(force=True)
    mode = data.get("mode", "paired")
    prefix = STATE.get("prefix")
    if not prefix:
        return jsonify({"ok": False, "error": "Go to Step 1 first."}), 400

    KMER_OUT_DIR.mkdir(parents=True, exist_ok=True)
    r1 = STATE["r1"]
    r2 = STATE["r2"]
    combined = STATE["combined"]

    if mode == "paired":
        input_prep = f"cat {shlex.quote(r1)} {shlex.quote(r2)} > {shlex.quote(combined)} && "
        input_file = combined
    else:
        input_prep = ""
        input_file = r1

    out_prefix = KMER_OUT_DIR / "kmergenie_out"
    cmd = input_prep + conda_cmd("kmergenie_env", f"kmergenie {shlex.quote(input_file)} -o {shlex.quote(str(out_prefix))}")

    job_id = start_job(cmd, ROOT, {"step": "kmergenie"})
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/run_cogram/start", methods=["POST"])
def run_cogram_start():
    data = request.get_json(force=True)
    use_best_k = bool(data.get("use_best_k", True))

    k = STATE.get("best_k") if use_best_k else int(data.get("k", 21))
    if not k:
        k = 21

    genome_file = STATE.get("genome_file")
    reads_file = STATE["combined"] if Path(STATE["combined"] or "").exists() else STATE["r1"]

    cmd = conda_cmd(
        "cogram_env",
        " ".join([
            "python main.py",
            f"--k {k}",
            f"--reads_file {shlex.quote(reads_file)}",
            f"--genome_file {shlex.quote(genome_file)}",
        ]),
    )

    job_id = start_job(cmd, COGRAM_DIR, {"step": "cogram", "k": k})
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/job/<job_id>")
def job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "Job not found"}), 404
        response = {
            "ok": True,
            "done": job["done"],
            "returncode": job["returncode"],
            "log": job["log"],
            "meta": job.get("meta", {}),
        }

    if response["done"] and response["returncode"] == 0:
        step = response["meta"].get("step")
        if step == "kmergenie":
            best_k, rows = parse_kmergenie_dat(KMER_OUT_DIR / "kmergenie_out.dat")
            STATE["best_k"] = best_k
            response["result"] = {"best_k": best_k, "table": rows, "report_url": "/kmergenie-report"}
        elif step == "cogram":
            out = response["log"]
            response["result"] = {
                "k": response["meta"].get("k"),
                "total_contigs": _extract_int(out, r"Total contigs:\s+(\d+)"),
                "total_assembled_length": _extract_int(out, r"Total assembled length:\s+(\d+)"),
                "longest_contig": _extract_int(out, r"Longest contig:\s+(\d+)"),
                "n50": _extract_int(out, r"N50:\s+(\d+)"),
                "approx_coverage": _extract_float(out, r"Approx coverage:\s+([0-9.]+)"),
                "contigs_fasta": str(COGRAM_DIR / "contigs.fasta"),
            }

    return jsonify(response)


@app.route("/kmergenie-report")
def kmergenie_report():
    return send_from_directory(KMER_OUT_DIR, "kmergenie_out_report.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
