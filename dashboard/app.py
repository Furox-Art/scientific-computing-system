import math
import random
import sys
from pathlib import Path

import streamlit as st

# Add src to path if running directly from dashboard/ folder
root_path = Path(__file__).parent.parent
if str(root_path / "src") not in sys.path:
    sys.path.append(str(root_path / "src"))

try:
    from cds import __version__
    from cds.data_analysis import plot_line
    from cds.hypothesis import Domain, generate_hypotheses
    from cds.ml import MLP, Layer
    from cds.quantum import QuantumCircuit, hadamard, simulate
except ImportError:
    st.error(
        "Could not import the 'cds' package. Ensure the 'src' directory is in your PYTHONPATH."
    )
    st.stop()

st.set_page_config(page_title="System Interactive Dashboard", layout="wide")

st.title("Cognitive Discovery System")
st.markdown("""
Welcome to the interactive showcase of the **System**.
Everything you see here is powered by **Pure Python** algorithms, built from scratch without NumPy, SciPy, or other heavy numerical dependencies.
""")

tabs = st.tabs(["Hypothesis Engine", "Quantum Lab", "Neural Network", "Math & Stats"])

# --- Tab 1: Hypothesis Engine ---
with tabs[0]:
    st.header("Structured Hypothesis Generation")
    st.write(
        "Generate scientific hypotheses with explicit reasoning, assumptions, and predictions."
    )
    col1, col2 = st.columns([2, 1])

    with col1:
        question = st.text_input(
            "Enter a Research Question:", value="What causes the Hubble tension?"
        )
        domain_choice = st.selectbox(
            "Select Domain:", ["Cosmology", "Physics", "Mathematics", "General Science"]
        )
        num_hypo = st.slider("Number of Hypotheses:", 1, 5, 2)

        if st.button("Generate Hypotheses", type="primary"):
            if not question.strip():
                st.error("Please enter a research question.")
            else:
                with st.spinner("Reasoning..."):
                    try:
                        dom = Domain(domain_choice.lower().replace(" ", "_"))
                        hypos = generate_hypotheses(question, domain=dom, n=num_hypo)

                        if not hypos:
                            st.warning("No hypotheses were generated. Try a different question.")
                            st.session_state.pop("hypotheses", None)
                        else:
                            # Persist across reruns so the result survives the
                            # next widget interaction (Streamlit reruns the whole
                            # script on every input change).
                            st.session_state["hypotheses"] = hypos
                    except Exception as e:
                        st.error(f"Error generating hypotheses: {str(e)}")
                        st.session_state.pop("hypotheses", None)

        # Render whatever is cached (survives unrelated widget reruns).
        cached_hypos = st.session_state.get("hypotheses")
        if cached_hypos:
            st.caption(f"Showing {len(cached_hypos)} hypothesis(ies).")
            for h in cached_hypos:
                with st.expander(f"ID: {h.id} - {h.statement[:50]}...", expanded=True):
                    st.markdown(h.to_markdown())

    with col2:
        st.info(
            "**Discovery Tip:** Generated hypotheses are structured with explicit assumptions and measurable predictions, ready for statistical testing."
        )

# --- Tab 2: Quantum Lab ---
with tabs[1]:
    st.header("Quantum Circuit Simulator")
    st.write("Simulate quantum gates and measure state probabilities using pure linear algebra.")

    q_col1, q_col2 = st.columns([1, 2])

    with q_col1:
        st.subheader("Circuit Builder")
        apply_h = st.checkbox("Apply Hadamard (H) to Qubit 0", value=True)
        st.caption("The Hadamard gate creates superposition: |0⟩ → (|0⟩ + |1⟩)/√2")
        shots = st.select_slider(
            "Number of Shots (Measurements):", options=[100, 1000, 10000, 100000], value=1000
        )

        if st.button("Run Simulation", type="primary"):
            try:
                c = QuantumCircuit()
                if apply_h:
                    c.add(hadamard())

                with st.spinner("Simulating..."):
                    counts = simulate(c, shots=shots)
                    st.session_state["q_counts"] = counts
                    st.success("Simulation Complete!")
            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")

    with q_col2:
        st.subheader("Measurement Statistics")
        if "q_counts" in st.session_state:
            counts = st.session_state["q_counts"]
            # ``simulate`` returns int-keyed counts {0: n, 1: n}. Relabel to
            # Dirac notation so the bar chart reads |0> / |1> instead of bare
            # integers, and sort by bitstring for a stable left-to-right order.
            labeled = {f"|{k}⟩": v for k, v in sorted(counts.items())}
            st.bar_chart(labeled)
            st.json(labeled)
        else:
            st.info("Construct a circuit and click 'Run Simulation' to see results.")

# --- Tab 3: Neural Network ---
with tabs[2]:
    st.header("MLP: Multi-Layer Perceptron")
    st.markdown(
        "Watch a Pure Python Neural Network learn the **XOR logic** in real-time using backpropagation."
    )

    if st.button("Train XOR Model", type="primary"):
        try:
            # Training data
            X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
            y = [[0.0], [1.0], [1.0], [0.0]]  # XOR

            net = MLP([Layer(2, 4, activation="relu"), Layer(4, 1, activation="sigmoid")])

            progress_bar = st.progress(0)
            status_text = st.empty()

            # Simple training loop with UI updates
            losses = []
            for epoch in range(1, 101):
                train_res = net.train(X, y, epochs=5, lr=0.1)
                losses.append(train_res["final_loss"])
                progress_bar.progress(epoch)
                status_text.text(f"Epoch {epoch * 5}/500 - Loss: {train_res['final_loss']:.6f}")

            st.success("Model Trained!")
            st.line_chart(losses, use_container_width=True)

            st.subheader("Predictions")
            cols = st.columns(4)
            for i, xi in enumerate(X):
                pred = net.predict(xi)[0]
                cols[i].metric(
                    f"Input {xi}",
                    f"{pred:.4f}",
                    delta="Expected 1" if y[i][0] == 1.0 else "Expected 0",
                )
        except Exception as e:
            st.error(f"Training failed: {str(e)}")

# --- Tab 4: Math & Stats ---
with tabs[3]:
    st.header("Numerical Analysis Showcase")
    st.write("Direct implementation of statistical tests and terminal-friendly visualizations.")

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.subheader("Welch's t-test")
        st.markdown("Comparing two distributions with unequal variances (Pure Python).")

        mu1 = st.slider("Mean 1", 50.0, 100.0, 70.0)
        mu2 = st.slider("Mean 2", 50.0, 100.0, 75.0)
        seed = st.slider(
            "Random Seed",
            0,
            100,
            42,
            help="Pin the seed so tweaking the means doesn't redraw the samples.",
        )

        # Seeded RNG: without this, every slider movement regenerated fresh
        # samples and the p-value jittered even when only the seed-irrelevant
        # widget changed. Pinning the seed keeps the comparison reproducible.
        rng = random.Random(seed)
        data1 = [rng.gauss(mu1, 5) for _ in range(50)]
        data2 = [rng.gauss(mu2, 8) for _ in range(50)]

        try:
            from cds.stats import TestResult, two_sample_ttest

            res: TestResult = two_sample_ttest(data1, data2, equal_var=False)

            st.metric("p-value", f"{res.p_value:.4f}")
            if res.p_value < 0.05:
                st.success("Significant Difference Found!")
            else:
                st.warning("No Significant Difference.")
        except Exception as e:
            st.error(f"Statistical test failed: {str(e)}")

    with m_col2:
        st.subheader("ASCII Visualization Engine")
        st.markdown("The underlying engine generating terminal-friendly plots.")
        try:
            wave_data = [math.sin(x * 0.3) for x in range(30)]
            st.text(plot_line(wave_data, title="Pure Python Sine Wave"))
        except Exception as e:
            st.error(f"Visualization failed: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.title("System Status")
st.sidebar.success(f"Core: v{__version__}")
st.sidebar.info("Dependencies: Standard Library + Typer/Rich/Pydantic")
st.sidebar.markdown("""
[View Source on GitHub](https://github.com/Furox88/scientific-computing-system)
""")
