#!/usr/bin/env python3
"""Script pour nettoyer la mémoire GPU utilisée par ColPali."""

import logging
import os
import signal
import subprocess

import psutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def find_colpali_processes() -> list[int]:
    """Trouve tous les processus qui pourraient utiliser ColPali.

    Returns:
        list[int]: Liste des PIDs des processus ColPali trouvés.
    """
    colpali_pids = []

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline", [])
            if cmdline and any("colpali" in str(arg).lower() for arg in cmdline):
                colpali_pids.append(proc.info["pid"])
                logger.info(f"Trouvé processus ColPali: PID {proc.info['pid']} - {' '.join(cmdline[:3])}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return colpali_pids


def kill_processes(pids: list[int], force: bool = False) -> None:
    """Termine les processus donnés.

    Args:
        pids (list[int]): Liste des PIDs à terminer.
        force (bool): Si True, utilise SIGKILL au lieu de SIGTERM.
    """
    for pid in pids:
        try:
            if force:
                os.kill(pid, signal.SIGKILL)
                logger.info(f"Processus {pid} tué (SIGKILL)")
            else:
                os.kill(pid, signal.SIGTERM)
                logger.info(f"Signal SIGTERM envoyé au processus {pid}")
        except ProcessLookupError:
            logger.warning(f"Processus {pid} déjà terminé")
        except PermissionError:
            logger.error(f"Permission refusée pour tuer le processus {pid}")


def clean_gpu_memory() -> None:
    """Nettoie la mémoire GPU en utilisant PyTorch et nvidia-smi."""
    try:
        # Essayer d'importer torch pour nettoyer
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("✅ Cache CUDA vidé avec PyTorch")
    except ImportError:
        logger.warning("PyTorch non disponible pour le nettoyage")

    # Utiliser nvidia-smi si disponible
    try:
        # D'abord, obtenir les infos GPU
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            logger.info("Processus GPU actuels:")
            for line in result.stdout.strip().split("\n"):
                if line:
                    logger.info(f"  {line}")

        # Essayer de reset (nécessite sudo)
        if os.geteuid() == 0:  # Si root
            subprocess.run(["nvidia-smi", "--gpu-reset"], capture_output=True)
            logger.info("✅ GPU reset effectué")
        else:
            logger.info("ℹ️ Exécutez avec sudo pour un reset GPU complet")

    except FileNotFoundError:
        logger.warning("nvidia-smi non trouvé")
    except Exception as e:
        logger.error(f"Erreur avec nvidia-smi: {e}")


def main() -> None:
    """Fonction principale du script de nettoyage GPU."""
    import argparse

    parser = argparse.ArgumentParser(description="Nettoie la mémoire GPU utilisée par ColPali")
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force la terminaison des processus (SIGKILL)",
    )
    parser.add_argument(
        "--kill-all",
        "-k",
        action="store_true",
        help="Tue tous les processus ColPali trouvés",
    )

    args = parser.parse_args()

    logger.info("🔍 Recherche des processus ColPali...")

    # Trouver les processus
    colpali_pids = find_colpali_processes()

    if not colpali_pids:
        logger.info("✅ Aucun processus ColPali trouvé")
    else:
        logger.info(f"📋 {len(colpali_pids)} processus ColPali trouvé(s)")

        if args.kill_all:
            logger.info("⚠️ Terminaison des processus...")
            kill_processes(colpali_pids, force=args.force)

    # Nettoyer la mémoire GPU
    logger.info("\n🧹 Nettoyage de la mémoire GPU...")
    clean_gpu_memory()

    # Afficher l'état final
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("\n📊 État GPU après nettoyage:")
            # Extraire juste les lignes importantes
            lines = result.stdout.split("\n")
            for i, line in enumerate(lines):
                if "MiB" in line and ("Default" in line or "Processes:" in lines[max(0, i - 1)]):
                    logger.info(line)
    except subprocess.SubprocessError as e:
        logger.warning(f"Impossible d'obtenir l'état GPU: {e}")
    except Exception as e:
        logger.error(f"Erreur inattendue lors de l'affichage de l'état GPU: {e}")

    logger.info("\n✅ Nettoyage terminé")


if __name__ == "__main__":
    main()
