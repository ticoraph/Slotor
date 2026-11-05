"""Gradio front end: wires widgets to ``slotor.service``; no business logic lives here (D2)."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import gradio as gr

from slotor import service
from slotor.catalog import Catalog

BUDGETS = {"10 s": 10.0, "1 min": 60.0, "5 min": 300.0}
POLL_SECONDS = 0.5
_USAGE_HEADERS = ["Référence", "Pièce", "Utilisées", "Disponibles", "Restantes"]
_EMPTY_OUTPUT = ("", "", "", [], "", None, None)
EFFORTS = {"rapide": "fast", "normal": "normal", "approfondie": "thorough"}
PURCHASE_BUDGETS = {"1 min": 60.0, "5 min": 300.0, "15 min": 900.0}


def build_ui(catalog: Catalog | None = None) -> gr.Blocks:
    catalog = catalog or service.default_catalog()
    jobs: dict[str, service.OptimizationJob] = {}
    export_dir = Path(tempfile.mkdtemp(prefix="slotor-"))

    def on_import(text: str):
        try:
            inventory = service.import_ta_export(catalog, text or "")
        except ValueError as exc:
            gr.Warning(str(exc))
            return gr.update(), f"Import refusé : {exc}"
        return service.inventory_rows(catalog, inventory), service.import_summary(
            catalog, inventory
        )

    def on_run(
        width,
        height,
        margin,
        budget_label,
        seed,
        top_k,
        lane,
        bridges,
        cap,
        rows,
        request: gr.Request,
    ):
        key = request.session_hash
        previous = jobs.get(key)
        if previous is not None:
            previous.cancel()
        try:
            req = service.OptimizeRequest(
                surface_width_mm=float(width),
                surface_height_mm=float(height),
                margin_mm=float(margin or 0),
                inventory=service.inventory_from_rows(catalog, rows),
                time_budget_s=BUDGETS[budget_label],
                seed=int(seed or 0),
                top_k=int(top_k),
                lane_spacing_mm=float(lane) if lane else None,
                allow_bridges=bool(bridges),
                max_elevated=int(cap) if bridges and cap not in (None, "") else None,
            )
            problem = service.build_problem(catalog, req)
        except (ValueError, TypeError) as exc:
            gr.Warning(str(exc))
            yield (f"Entrée invalide : {exc}", *_EMPTY_OUTPUT[1:])
            return
        job = service.OptimizationJob(
            problem, seed=req.seed, time_budget_s=req.time_budget_s, top_k=req.top_k
        )
        jobs[key] = job
        job.start()
        while True:
            snapshot = job.snapshot()
            shown = service.present(problem, snapshot, req.lane_spacing_mm)
            finished = snapshot.status != "running"
            json_path, svg_path = (
                service.write_exports(shown, export_dir) if finished else (None, None)
            )
            yield (
                shown.status,
                shown.summary,
                shown.drawings_html,
                shown.usage,
                shown.bom,
                json_path,
                svg_path,
            )
            if finished:
                return
            time.sleep(POLL_SECONDS)

    def on_stop(request: gr.Request):
        job = jobs.get(request.session_hash)
        if job is not None:
            job.cancel()

    purchase_jobs: dict[str, object] = {}

    def on_buy(
        width,
        height,
        margin,
        inventory_rows,
        price_rows,
        budget,
        support,
        bridges,
        cap,
        effort_label,
        time_label,
        request: gr.Request,
    ):
        key = request.session_hash
        previous = purchase_jobs.get(key)
        if previous is not None:
            previous.cancel()
        try:
            req = service.PurchaseRequest(
                surface_width_mm=float(width),
                surface_height_mm=float(height),
                margin_mm=float(margin or 0),
                inventory=service.inventory_from_rows(catalog, inventory_rows),
                prices=service.prices_from_rows(price_rows),
                budget=float(budget or 0),
                support_price=float(support or 0),
                allow_bridges=bool(bridges),
                max_elevated=int(cap) if bridges and cap not in (None, "") else None,
                effort=EFFORTS[effort_label],
                time_budget_s=PURCHASE_BUDGETS[time_label],
            )
            problem, options = service.build_purchase(catalog, req)
        except (ValueError, TypeError) as exc:
            gr.Warning(str(exc))
            yield (f"Entrée invalide : {exc}", "", [], "", "", "")
            return
        job = service.create_purchase_job(problem, options, req)
        purchase_jobs[key] = job
        job.start()
        while True:
            snapshot = job.snapshot()
            shown = service.present_purchase(problem, snapshot, req)
            yield (
                shown.status, shown.summary, shown.steps, shown.shopping_list,
                shown.drawings_html, shown.bom,
            )  # fmt: skip
            if snapshot.status != "running":
                return
            time.sleep(POLL_SECONDS)

    def on_buy_stop(request: gr.Request):
        job = purchase_jobs.get(request.session_hash)
        if job is not None:
            job.cancel()

    with gr.Blocks(title="Slotor") as demo:
        gr.Markdown(
            "# Slotor — le plus long circuit fermé\n"
            "Projet indépendant, non affilié à Carrera. Saisissez la surface et votre inventaire : "
            "chaque circuit affiché a été validé (inventaire, surface, fermeture, collisions)."
        )
        with gr.Tabs():
            with gr.Tab("Optimiser"), gr.Row():
                with gr.Column(scale=1):
                    width = gr.Number(3000, label="Largeur de la surface (mm)")
                    height = gr.Number(1440, label="Profondeur de la surface (mm)")
                    margin = gr.Number(0, label="Marge (mm)")
                    budget = gr.Dropdown(list(BUDGETS), value="10 s", label="Budget de calcul")
                    seed = gr.Number(0, precision=0, label="Graine (reproductibilité)")
                    top_k = gr.Slider(1, 5, value=3, step=1, label="Nombre de circuits proposés")
                    lane = gr.Number(
                        None,
                        label="Entraxe des fentes (mm), optionnel : active la longueur des voies",
                    )
                    bridges = gr.Checkbox(
                        False,
                        label=(
                            "Autoriser les ponts (expérimental : 3 éléments pour monter, "
                            "au moins 1 de tablier, 3 pour descendre ; supports à acheter à part)"
                        ),
                    )
                    cap = gr.Number(
                        None,
                        precision=0,
                        label="Pièces surélevées au maximum (vide = illimité ; supports à prévoir)",
                    )
                    inventory = gr.Dataframe(
                        value=service.inventory_rows(catalog),
                        headers=["Référence", "Pièce", "Quantité"],
                        datatype=["str", "str", "number"],
                        interactive=True,
                        static_columns=[0, 1],
                        column_widths=["30%", "45%", "25%"],
                        wrap=True,
                        label="Inventaire (quantité modifiable)",
                        type="array",
                    )
                    ta_text = gr.Textbox(
                        lines=6, label="Ou collez l'export texte de Track Architect"
                    )
                    import_btn = gr.Button("Importer l'export")
                    import_msg = gr.Markdown()
                with gr.Column(scale=2):
                    with gr.Row():
                        run_btn = gr.Button("Optimiser", variant="primary")
                        stop_btn = gr.Button("Arrêter")
                    status = gr.Markdown()
                    summary = gr.Markdown()
                    drawings = gr.HTML()
                    usage = gr.Dataframe(
                        headers=_USAGE_HEADERS, interactive=False, label="Pièces utilisées"
                    )
                    bom = gr.Textbox(label="Nomenclature", lines=8, interactive=False)
                    with gr.Row():
                        json_file = gr.File(label="Export JSON")
                        svg_file = gr.File(label="Export SVG")

            with gr.Tab("Que racheter ?"):
                gr.Markdown(
                    "Combien de longueur gagne-t-on avec un budget de rails ? Saisissez le "
                    "**prix de chaque lot que vous envisagez d'acheter** (le catalogue n'en "
                    "contient pas ; laissez vide un lot écarté) et un budget dans la même unité. "
                    "Chaque longueur annoncée est celle d'un circuit validé."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        b_width = gr.Number(3000, label="Largeur de la surface (mm)")
                        b_height = gr.Number(1440, label="Profondeur de la surface (mm)")
                        b_margin = gr.Number(0, label="Marge (mm)")
                        b_budget = gr.Number(100, label="Budget d'achat")
                        b_support = gr.Number(
                            0, label="Prix d'un support de pont (compté par élément surélevé)"
                        )
                        b_bridges = gr.Checkbox(False, label="Autoriser les ponts (expérimental)")
                        b_cap = gr.Number(
                            None,
                            precision=0,
                            label="Supports possédés = éléments surélevés max (vide : à acheter)",
                        )
                        b_effort = gr.Dropdown(
                            list(EFFORTS), value="normal", label="Précision de l'estimation"
                        )
                        b_time = gr.Dropdown(
                            list(PURCHASE_BUDGETS), value="5 min", label="Temps de calcul maximal"
                        )
                        b_inventory = gr.Dataframe(
                            value=service.inventory_rows(catalog),
                            headers=["Référence", "Pièce", "Quantité"],
                            datatype=["str", "str", "number"],
                            interactive=True,
                            static_columns=[0, 1],
                            column_widths=["30%", "45%", "25%"],
                            wrap=True,
                            label="Ce que vous possédez déjà (quantité modifiable)",
                            type="array",
                        )
                        b_prices = gr.Dataframe(
                            value=service.purchase_price_rows(catalog),
                            headers=["Lot", "Nom", "Contenu", "Prix"],
                            datatype=["str", "str", "str", "number"],
                            interactive=True,
                            static_columns=[0, 1, 2],
                            wrap=True,
                            label="Prix des lots (à saisir)",
                            type="array",
                        )
                    with gr.Column(scale=2):
                        with gr.Row():
                            b_run = gr.Button("Planifier les achats", variant="primary")
                            b_stop = gr.Button("Arrêter")
                        b_status = gr.Markdown()
                        b_summary = gr.Markdown()
                        b_steps = gr.Dataframe(
                            headers=service.PURCHASE_HEADERS,
                            interactive=False,
                            label="Achats, dans l'ordre",
                        )
                        b_list = gr.Textbox(label="Liste de courses", lines=6, interactive=False)
                        b_drawings = gr.HTML()
                        b_bom = gr.Textbox(
                            label="Nomenclature du circuit obtenu", lines=8, interactive=False
                        )

        import_btn.click(on_import, inputs=[ta_text], outputs=[inventory, import_msg])
        run_btn.click(
            on_run,
            inputs=[width, height, margin, budget, seed, top_k, lane, bridges, cap, inventory],
            outputs=[status, summary, drawings, usage, bom, json_file, svg_file],
        )
        stop_btn.click(on_stop, inputs=None, outputs=None)
        b_run.click(
            on_buy,
            inputs=[
                b_width, b_height, b_margin, b_inventory, b_prices, b_budget, b_support,
                b_bridges, b_cap, b_effort, b_time,
            ],
            outputs=[b_status, b_summary, b_steps, b_list, b_drawings, b_bom],
        )  # fmt: skip
        b_stop.click(on_buy_stop, inputs=None, outputs=None)
    return demo


def main() -> None:
    """Serve on the loopback interface only: this is a local tool, never exposed."""
    build_ui().queue().launch(server_name="127.0.0.1", inbrowser=True, share=False)
