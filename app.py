                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if not GROQ_API_KEY:
                st.warning(
                    "Chat is not enabled. Add GROQ_API_KEY to Streamlit Secrets "
                    "to activate the RAG assistant."
                )
            elif Groq is None or SentenceTransformer is None:
                st.warning(
                    "Chat dependencies are unavailable. Install groq and "
                    "sentence-transformers from requirements.txt."
                )
            else:
                user_question = st.chat_input(
                    "Ask about MQL rules, fit score, engagement, tiers, or this dataset..."
                )
                if user_question:
                    st.session_state.messages.append({
                        "role": "user",
                        "content": user_question,
                    })
                    with st.chat_message("user"):
                        st.markdown(user_question)

                    try:
                        backend = load_rag_backend()
                        retrieved = retrieve(user_question, backend)
                        retrieved_context = "\n\n".join(
                            f"[{item['title']}] {item['text']}"
                            for item in retrieved
                        )
                        dataset_context = build_dataset_summary_context(scored_df)

                        if not retrieved_context:
                            retrieved_context = REFUSAL

                        system_prompt = f"""
You are an artifact-grounded RevOps assistant for a deterministic B2B group
insurance lead-prioritization framework.

Use ONLY the supplied knowledge-base context for claims about the documented
scoring methodology. Use the active dataset metrics for questions about the
currently selected dataset. Do not invent scoring rules, thresholds, data
fields, customer facts, or results.

The framework is deterministic and heuristic, not a predictive conversion model.
Do not describe it as an ML prediction model.

If the supplied context does not support an answer, say exactly:
"{REFUSAL}"

Cite relevant knowledge chunks using their title in square brackets, for example
[Model B Overview]. Keep answers concise but useful.

ACTIVE DATASET:
{dataset_context}

RETRIEVED KNOWLEDGE-BASE CONTEXT:
{retrieved_context}
""".strip()

                        with st.chat_message("assistant"):
                            with st.spinner("Retrieving evidence and generating answer..."):
                                response = backend[3].chat.completions.create(
                                    model="llama-3.1-8b-instant",
                                    messages=[
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": user_question},
                                    ],
                                    temperature=0.2,
                                    max_tokens=MAX_TOKENS_DEFAULT,
                                )
                                answer = response.choices[0].message.content or REFUSAL
                            st.markdown(answer)
                            with st.expander("View retrieved grounding context"):
                                if retrieved:
                                    for item in retrieved:
                                        st.markdown(
                                            f"**{item['title']}** "
                                            f"(similarity: {item['raw_score']:.3f})"
                                        )
                                        st.write(item["text"])
                                else:
                                    st.write("No relevant knowledge chunks passed the similarity threshold.")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                        })

                    except Exception as exc:
                        st.error(f"RAG assistant error: {exc}")

        with tab_rules:
            st.subheader("Model B scoring configuration")
            st.markdown(
                """
**Important:** Model B is a deterministic, heuristic prioritization framework.
It is not designed to predict whether an account will convert.

- Email events receive category-based points and time decay.
- Webinar actions contribute engagement points.
- Contact engagement is aggregated to account engagement.
- Fit score uses renewal proximity, estimated premium, industry fit, carrier, and broker relationship.
- Composite score uses the configured formula: `0.4 × fit_score + 0.6 × account_engagement + 0.8 × broker_bonus`.
- Qualification tiers and archetypes are assigned through explicit business rules.
- Proposal presence is used for the `sql_converted` output flag in this implementation.
                """
            )
            st.caption(
                "The evaluation date is fixed in this version at "
                f"{EVALUATION_DATE.strftime('%Y-%m-%d')}. Email decay and renewal "
                "urgency are calculated relative to that date."
            )

else:
    st.warning(
        "No scored dataset is currently available. Select the default dataset "
        "after adding the six repository CSV files, or choose 'Upload my own data'."
    )

st.divider()
st.caption(
    "RevOps Lead Scoring Studio | Deterministic Model B scoring + knowledge-grounded RAG"
)
