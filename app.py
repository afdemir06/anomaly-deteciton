import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

API_BASE_URL_DEFAULT = "http://localhost:8000"
HEADERS = {"ngrok-skip-browser-warning": "true"}

st.set_page_config(
    page_title="TimeGuard — Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stMetric label { font-size: 0.85rem !important; }
    div[data-testid="stSidebar"] { border-right: 1px solid #333; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────

with st.sidebar:
    st.header("TimeGuard")
    st.caption("Anomaly Detection System")

    st.divider()
    api_url = st.text_input(
        "API URL",
        value=st.session_state.get("api_url", API_BASE_URL_DEFAULT),
        help="Colab/ngrok URL or local API address",
    )
    st.session_state["api_url"] = api_url

    uploaded_file = st.file_uploader("Select a CSV file", type=["csv"])
    run_id_input = st.text_input("Run ID (optional)")

    col1, col2 = st.columns(2)
    with col1:
        train_clicked = st.button("Train", use_container_width=True)
    with col2:
        detect_clicked = st.button("Fetch Results", use_container_width=True)

    if train_clicked:
        if uploaded_file is None:
            st.warning("Please upload a CSV file first")
        else:
            with st.spinner("Training... please wait"):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "text/csv",
                        )
                    }
                    response = requests.post(
                        f"{st.session_state.get('api_url', API_BASE_URL_DEFAULT)}/train",
                        files=files, headers=HEADERS, timeout=None,
                    )
                    if response.status_code == 200:
                        st.session_state["train_result"] = response.json()
                        st.session_state["run_id"] = response.json()["run_id"]
                        st.success("Training completed!")
                    else:
                        detail = response.json().get("detail", "Unknown error")
                        st.error(f"Training error: {detail}")
                except requests.ConnectionError:
                    st.error("Cannot connect to API. Make sure the server is running.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    if detect_clicked:
        with st.spinner("Fetching results..."):
            try:
                params = {}
                if run_id_input.strip():
                    params["run_id"] = run_id_input.strip()
                response = requests.post(
                    f"{st.session_state.get('api_url', API_BASE_URL_DEFAULT)}/detect",
                    params=params, headers=HEADERS, timeout=None,
                )
                if response.status_code == 200:
                    st.session_state["detect_result"] = response.json()
                    st.session_state["run_id"] = response.json()["run_id"]
                    st.success("Results fetched!")
                else:
                    detail = response.json().get("detail", "Unknown error")
                    st.error(f"Error: {detail}")
            except requests.ConnectionError:
                st.error("Cannot connect to API. Make sure the server is running.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# ── Tabs ───────────────────────────────────────────────────────

tab_train, tab_results, tab_explain, tab_model = st.tabs(
    ["Training", "Results", "Explanations", "Model Info"]
)

# ── Tab 1: Training ──────────────────────────────────────────

with tab_train:
    if "train_result" not in st.session_state:
        st.info("No training yet. Upload a CSV file in the sidebar and click 'Train'.")
    else:
        result = st.session_state["train_result"]

        st.subheader("Training Result")
        st.metric("Run ID", result["run_id"])

        st.divider()
        st.subheader("Model Metrics")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Isolation Forest**")
            st.metric("Anomalies", result["if_metrics"]["n_anomalies"])
            st.metric(
                "Anomaly Rate",
                f"{result['if_metrics']['anomaly_rate']:.2%}",
            )

        with c2:
            st.markdown("**DBSCAN**")
            st.metric("Anomalies", result["dbscan_metrics"]["n_anomalies"])
            st.metric(
                "Anomaly Rate",
                f"{result['dbscan_metrics']['anomaly_rate']:.2%}",
            )

        with c3:
            st.markdown("**LSTM Autoencoder**")
            st.metric("Anomalies", result["lstm_metrics"]["n_anomalies"])
            st.metric(
                "Anomaly Rate",
                f"{result['lstm_metrics']['anomaly_rate']:.2%}",
            )

        st.divider()
        st.subheader("Ensemble Metrics")
        ec1, ec2 = st.columns(2)
        with ec1:
            st.metric("Total Anomalies", result["ensemble_metrics"]["n_anomalies"])
        with ec2:
            st.metric(
                "Anomaly Rate",
                f"{result['ensemble_metrics']['anomaly_rate']:.2%}",
            )

# ── Tab 2: Results ──────────────────────────────────────────

with tab_results:
    if "detect_result" not in st.session_state:
        st.info(
            "Train a model first or click 'Fetch Results' in the sidebar."
        )
    else:
        result = st.session_state["detect_result"]

        if not result.get("timestamps") or not result.get("feature_values"):
            st.warning("No time series data available for this run.")
        else:
            df = pd.DataFrame(result["feature_values"])
            df.insert(0, "timestamp", result["timestamps"])
            df["is_anomaly"] = result["is_anomaly"]
            df["anomaly_score"] = result["anomaly_score"]
            df["if_vote"] = result["if_votes"]
            df["dbscan_vote"] = result["dbscan_votes"]
            df["lstm_vote"] = result["lstm_votes"]

            st.subheader("Time Series Plot")

            feature_cols = [c for c in df.columns if c not in (
                "timestamp", "is_anomaly", "anomaly_score",
                "if_vote", "dbscan_vote", "lstm_vote",
            )]

            fig = go.Figure()
            for col in feature_cols:
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df[col],
                        name=col,
                        mode="lines",
                    )
                )

            anomaly_times = df.loc[df["is_anomaly"] == True, "timestamp"]
            for t in anomaly_times:
                fig.add_vline(
                    x=t,
                    line_dash="dash",
                    line_color="#E53E3E",
                    opacity=0.6,
                )

            fig.update_layout(
                xaxis_title="Timestamp",
                yaxis_title="Value",
                legend_title="Features",
                margin=dict(l=0, r=0, t=30, b=0),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Prediction Details")
            show_only_anomalies = st.checkbox("Show anomalies only")

            display_df = df[["timestamp", "is_anomaly", "anomaly_score",
                             "if_vote", "dbscan_vote", "lstm_vote"]].copy()
            display_df["timestamp"] = display_df["timestamp"].astype(str)

            if show_only_anomalies:
                display_df = display_df[display_df["is_anomaly"] == True]

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            predictions = np.array(result["predictions"])
            if_votes = np.array(result["if_votes"])
            dbscan_votes = np.array(result["dbscan_votes"])
            lstm_votes = np.array(result["lstm_votes"])
            ground_truth = result.get("ground_truth")
            has_ground_truth = (
                ground_truth is not None
                and any(v is not None for v in ground_truth)
            )

            def _anomaly_metrics(model_preds, ref):
                tp = int(np.sum((model_preds == 1) & (ref == 1)))
                fp = int(np.sum((model_preds == 1) & (ref == 0)))
                fn = int(np.sum((model_preds == 0) & (ref == 1)))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )
                return precision, recall, f1

            if has_ground_truth:
                gt = np.array(ground_truth, dtype=int)

                st.subheader("Confusion Matrix (Ensemble vs Ground Truth)")
                tp = int(np.sum((predictions == 1) & (gt == 1)))
                fp = int(np.sum((predictions == 1) & (gt == 0)))
                fn = int(np.sum((predictions == 0) & (gt == 1)))
                tn = int(np.sum((predictions == 0) & (gt == 0)))
                cm = np.array([[tn, fp], [fn, tp]])

                cm_fig = go.Figure(
                    data=go.Heatmap(
                        z=cm,
                        x=["Pred: Normal", "Pred: Anomaly"],
                        y=["True: Normal", "True: Anomaly"],
                        text=cm,
                        texttemplate="%{text}",
                        colorscale="Reds",
                        showscale=False,
                    )
                )
                cm_fig.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=300,
                )
                st.plotly_chart(
                    cm_fig, use_container_width=True, key="confusion_matrix"
                )

                st.divider()
                st.subheader(
                    "Anomaly Detection Metrics (vs Ground Truth)"
                )

                vote_arrays = {
                    "IF": if_votes,
                    "DBSCAN": dbscan_votes,
                    "LSTM": lstm_votes,
                    "Ensemble": predictions,
                }
                metrics_rows = []
                for name, preds in vote_arrays.items():
                    p, r, f = _anomaly_metrics(preds, gt)
                    metrics_rows.append(
                        {
                            "Model": name,
                            "Precision": round(p, 3),
                            "Recall": round(r, 3),
                            "F1": round(f, 3),
                        }
                    )
                metrics_df = pd.DataFrame(metrics_rows)
                st.dataframe(
                    metrics_df, use_container_width=True, hide_index=True
                )

            else:
                st.subheader("Model Agreement Heatmap")

                vote_arrays = {
                    "IF": if_votes,
                    "DBSCAN": dbscan_votes,
                    "LSTM": lstm_votes,
                    "Ensemble": predictions,
                }
                model_names = list(vote_arrays.keys())
                n_models = len(model_names)
                agreement = np.zeros((n_models, n_models))
                for i in range(n_models):
                    for j in range(n_models):
                        agreement[i, j] = np.mean(
                            vote_arrays[model_names[i]]
                            == vote_arrays[model_names[j]]
                        )

                heatmap_fig = go.Figure(
                    data=go.Heatmap(
                        z=agreement,
                        x=model_names,
                        y=model_names,
                        text=np.round(agreement * 100, 1),
                        texttemplate="%{text}%",
                        colorscale="Blues",
                        zmin=0,
                        zmax=1,
                    )
                )
                heatmap_fig.update_layout(
                    xaxis_title="Model",
                    yaxis_title="Model",
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=350,
                )
                st.plotly_chart(
                    heatmap_fig,
                    use_container_width=True,
                    key="agreement_heatmap",
                )

                st.divider()
                st.subheader("Anomaly Detection Metrics")

                ensemble_preds = predictions

                metrics_rows = []
                for name in ["IF", "DBSCAN", "LSTM"]:
                    p, r, f = _anomaly_metrics(
                        vote_arrays[name], ensemble_preds
                    )
                    metrics_rows.append(
                        {
                            "Model": name,
                            "Precision": round(p, 3),
                            "Recall": round(r, 3),
                            "F1": round(f, 3),
                        }
                    )

                metrics_df = pd.DataFrame(metrics_rows)
                st.dataframe(
                    metrics_df, use_container_width=True, hide_index=True
                )

# ── Tab 3: Explanations ─────────────────────────────────────

with tab_explain:
    current_run_id = st.session_state.get("run_id")
    if current_run_id is None:
        st.info("Train a model first.")
    else:
        with st.spinner("Fetching explanations..."):
            try:
                response = requests.get(
                    f"{st.session_state.get('api_url', API_BASE_URL_DEFAULT)}/detect/explain",
                    params={"run_id": current_run_id},
                    headers=HEADERS, timeout=120,
                )
                if response.status_code != 200:
                    detail = response.json().get("detail", "Unknown error!")
                    st.error(f"Error: {detail}")
                else:
                    data = response.json()
                    explanations = data.get("explanations", [])

                    if not explanations:
                        st.info("No anomalies to explain.")
                    else:
                        st.subheader(
                            f"Anomaly Explanations ({data['n_anomalies_explained']} total)"
                        )

                        for exp in explanations:
                            row_idx = exp.get("row_index", "?")
                            shap_data = exp.get("shap_explanation")
                            title = f"Row {row_idx}"

                            ts = exp.get("timestamp")
                            if shap_data and ts:
                                title = f"{ts}  —  Row {row_idx}"

                            with st.expander(title, expanded=False):
                                if (
                                    shap_data
                                    and shap_data.get("top_features")
                                ):
                                    features = shap_data["top_features"]
                                    values = [f["shap_value"] for f in features]
                                    names = [f["feature"] for f in features]
                                    colors = [
                                        "#E53E3E" if v > 0 else "#3182CE"
                                        for v in values
                                    ]

                                    bar_fig = go.Figure(
                                        go.Bar(
                                            x=values,
                                            y=names,
                                            orientation="h",
                                            marker_color=colors,
                                        )
                                    )
                                    bar_fig.update_layout(
                                        xaxis_title="SHAP Value",
                                        yaxis_title="Feature",
                                        margin=dict(l=0, r=0, t=10, b=0),
                                        height=max(250, len(names) * 28),
                                        yaxis=dict(autorange="reversed"),
                                    )
                                    st.plotly_chart(
                                        bar_fig, use_container_width=True,
                                        key=f"shap_bar_{row_idx}",
                                    )
                                else:
                                    msg = (
                                        shap_data.get("message", "")
                                        if shap_data
                                        else ""
                                    )
                                    st.info(
                                        msg
                                        or "IF did not flag this anomaly, no explanation available."
                                    )
            except requests.ConnectionError:
                st.error("Cannot connect to API.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# ── Tab 4: Model Info ───────────────────────────────────────

with tab_model:
    current_run_id = st.session_state.get("run_id")
    if current_run_id is None:
        st.info("Train a model first.")
    else:
        with st.spinner("Fetching model info..."):
            try:
                response = requests.get(
                    f"{st.session_state.get('api_url', API_BASE_URL_DEFAULT)}/model/info",
                    params={"run_id": current_run_id},
                    headers=HEADERS, timeout=30,
                )
                if response.status_code != 200:
                    detail = response.json().get("detail", "Unknown error")
                    st.error(f"Error: {detail}")
                else:
                    info = response.json()

                    st.subheader("General Info")
                    mc1, mc2 = st.columns(2)
                    with mc1:
                        st.metric("Run ID", info["id"])
                        st.metric("File Name", info["file_name"])
                        st.metric("Created At", str(info["created_at"]))
                    with mc2:
                        st.metric("Row Count", info["n_rows"])
                        st.metric("Feature Count", info["n_features"])

                    st.divider()
                    st.subheader("Features")
                    st.write(info["feature_cols"])

                    st.divider()
                    st.subheader("Model Metrics (JSON)")

                    cols = st.columns(3)
                    with cols[0]:
                        st.markdown("**Isolation Forest**")
                        st.json(info["if_metrics"])
                    with cols[1]:
                        st.markdown("**DBSCAN**")
                        st.json(info["dbscan_metrics"])
                    with cols[2]:
                        st.markdown("**LSTM Autoencoder**")
                        st.json(info["lstm_metrics"])

            except requests.ConnectionError:
                st.error("Cannot connect to API.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
