This code generates indicators like the ones in
Ghomi and Hurtado (2026): "RAUI: Uncertainty Indicators Built With Artificial Intelligence"
Banco de España Working paper n.2609

They are not exactly "replication codes" because we translated everything into English.
The aim is to make it easier for other researchers to apply our methodology to their data.

This code has been tested with ollama 0.12.9.
Newer versions of ollama may need adjustments in the code: there have been changes in how
the embedding models are prompted (new syntax, and error instead of truncate when the 
prompt is too long for the context of a specific model).

In any case, there are also newer models with supposedly better performance,
both for embedding and for quantification with LLM,
so an adaptation of the codes to use updated components is probably for the best.
