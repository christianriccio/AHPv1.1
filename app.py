import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import csv
import json
import math

# =========================================================
# UI TEXT PLACEHOLDER (da sostituire con il testo reale)
# =========================================================
TUTORIAL_PLACEHOLDER = """
### Come funziona questa valutazione

In questa intervista ti chiediamo di confrontare **a coppie** le fasi principali del ciclo di vita dei dati (Data Life Cycle).
Per ogni coppia ti chiederemo: *"Quale fase è più importante per noi?"* e *"Quanto è più importante (da 1 a 9)?"*.

- Valore 1 = le due fasi sono ugualmente importanti.
- Valore 9 = la fase scelta è estremamente più importante dell'altra.
- Usiamo questi confronti per calcolare i pesi dei criteri con il metodo AHP (Analytic Hierarchy Process).
- I pesi finali servono per valutare e confrontare diversi modelli di Data Lifecycle (DLM).

Qui sotto trovi una descrizione breve di ogni fase/criterio che confronterai.

---

#### 1. STARTING (Planning, Collection)
**Cosa è:** è la fase in cui si pianifica e si inizia a raccogliere i dati.
- *Planning*: definire obiettivi, responsabilità, risorse, dove finiranno i dati, come verranno gestiti e per quanto tempo.
- *Collection*: raccogliere/creare i dati da tutte le fonti rilevanti (sensoristica, banche dati pubbliche, citizen data, ecc.), in qualsiasi formato.

**Come interpretarla:** questa fase è centrale se per te conta soprattutto “mettere in piedi bene il progetto dati all’inizio”, cioè capire cosa serve, ottenere i dati giusti e impostare regole chiare fin dall’inizio.

---

#### 2. DATA ASSESSMENT (Preparation, Quality)
**Cosa è:** è la fase in cui i dati grezzi vengono resi utilizzabili e affidabili.
- *Preparation*: integrazione, pulizia, filtraggio, arricchimento dei dati; eliminare rumore, unificare formati e fonti.
- *Quality*: controllo della qualità nel tempo (accuratezza, completezza, coerenza, aggiornamento, assenza di errori).

**Come interpretarla:** questa fase è prioritaria se, per te, “avere dati puliti, affidabili e consistenti” è più critico di tutto il resto. Qui l’attenzione è sulla bontà tecnica del dato prima dell’uso.

---

#### 3. COMPUTATIONAL (Storage, Analysis, Visualization)
**Cosa è:** è la capacità operativa di lavorare sui dati.
- *Storage*: salvare i dati in modo sicuro, accessibile e scalabile (database, data lake, cloud, ecc.).
- *Analysis*: estrarre valore e conoscenza (analytics, ML/AI, modelli predittivi, ecc.).
- *Visualization*: comunicare i risultati in modo comprensibile (dashboard, report, grafici) per supportare decisioni.

**Come interpretarla:** questa fase è dominante se per te conta soprattutto “usare i dati per generare insight utili, prendere decisioni e comunicarle chiaramente”, con infrastrutture tecniche adeguate.

---

#### 4. DATA ADMINISTRATION (Share/Publish, Use & Reuse & Feedback, Governance)
**Cosa è:** è la gestione organizzativa e amministrativa dei dati nel loro ciclo di vita.
- *Use / Reuse / Feedback*: usare i dati, riusarli per nuovi scopi, raccogliere feedback dagli utenti e stakeholder.
- *Share / Publish*: rendere i dati disponibili e riutilizzabili (open data, scambio tra enti, pubblicazione controllata).
- *Governance*: regole, ruoli, responsabilità, processi decisionali su chi può fare cosa con i dati.

**Come interpretarla:** è centrale se per te la priorità è “far circolare i dati in modo controllato ma utile”, cioè massimizzare impatto, trasparenza, interoperabilità e accountability tra attori diversi (PA, cittadini, privati).

---

#### 5. SECURITY (Access, Protection)
**Cosa è:** è la dimensione sicurezza e tutela.
- *Access*: chi può vedere/ottenere cosa, con quali credenziali, attraverso quali canali.
- *Protection*: protezione di integrità, privacy e confidenzialità (controlli di accesso, criptazione, conformità normativa GDPR, sicurezza by design).

**Come interpretarla:** dai più importanza a questa fase se ritieni che la priorità è “proteggere dati sensibili e garantire accessi controllati”, anche a costo di rallentare la velocità operativa.

---

#### 6. ENDING (Archiving, End of Life)
**Cosa è:** è la gestione del fine vita dei dati.
- *Archiving*: conservare a lungo termine i dati storici in modo sicuro e rintracciabile, a basso costo.
- *End of Life*: eliminare i dati che non servono più o che non possono più essere legalmente trattenuti, garantendo che vengano davvero distrutti e non recuperabili.

**Come interpretarla:** è fondamentale se la tua priorità è la sostenibilità nel tempo: trattenere solo ciò che va mantenuto (per obblighi o valore storico) e cancellare in modo sicuro quello che deve essere eliminato.

---

### Come rispondere ai confronti
Per ogni coppia di fasi:
1. Scegli quale fase ritieni **più importante** per il nostro contesto decisionale.
2. Usa la scala da 1 a 9 per dire **quanto più importante** lo è.
   - 1 = pari importanza
   - 3 = moderatamente più importante
   - 5 = molto più importante
   - 7 = molto fortemente più importante
   - 9 = estremamente più importante

Non ci sono risposte giuste o sbagliate. Il tuo giudizio viene combinato con gli altri esperti in modo anonimo.
"""

# =========================================================
# FUNZIONI DI SUPPORTO I/O E STATO
# =========================================================

def init_session_state():
    """
    Inizializza lo stato di sessione se non è già presente.
    Questo serve per:
    - Backup/ripristino (JSON)
    """
    if "backup_state" not in st.session_state:
        st.session_state.backup_state = {
            "objective": "",
            "criteria": [],
            "alternatives": [],
            "performance_matrix": {},  # {alt: {crit: value}}
            "num_interviews": 1,
            "interviews": {},          # "0": {"pairwise": {...}}
        }


def load_alternatives_file(uploaded_file):
    """
    Caricamento del file con:
    - prima colonna = nome alternativa (DLM)
    - colonne successive = criteri numerici
    Supporta CSV con separatore autodetect, e Excel.
    """
    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower()

    # Excel
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        try:
            df = pd.read_excel(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Errore lettura Excel: {e}")
            return None

    # CSV con separatore sconosciuto
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        return df
    except Exception as e:
        st.warning(f"Tentativo autodetect CSV fallito: {e}")
        try:
            uploaded_file.seek(0)
            rawdata = uploaded_file.read(2048).decode('utf-8', errors='replace')
            uploaded_file.seek(0)
            dialect = csv.Sniffer().sniff(rawdata, delimiters=[',',';','\t','|'])
            sep = dialect.delimiter
            df = pd.read_csv(uploaded_file, sep=sep)
            return df
        except Exception as e2:
            st.error(f"Tentativo csv.Sniffer fallito: {e2}")
            return None


def download_json_button(label, data_dict, filename):
    """
    Bottone per scaricare un backup JSON dell'intero stato dell'esperimento
    (inclusi confronti degli esperti).
    """
    json_bytes = json.dumps(data_dict, indent=2).encode('utf-8')
    st.download_button(
        label=label,
        data=json_bytes,
        file_name=filename,
        mime='application/json'
    )


def try_load_backup_json(uploaded_json):
    """
    Carica un backup JSON precedentemente salvato.
    Restituisce il dict o None se fallisce.
    """
    try:
        raw = uploaded_json.read()
        data = json.loads(raw.decode('utf-8'))
        return data
    except Exception as e:
        st.error(f"Backup JSON non valido: {e}")
        return None


def matrix_to_dict(df):
    """Converte un DataFrame in dict serializzabile."""
    return {
        "index": list(df.index),
        "columns": list(df.columns),
        "data": df.values.tolist()
    }


def dict_to_matrix(d):
    """Converte dict serializzato da backup in DataFrame."""
    return pd.DataFrame(d["data"], index=d["index"], columns=d["columns"])


def save_current_state_to_session(objective,
                                  criteria_list,
                                  alternatives_list,
                                  perf_df,
                                  num_interviews,
                                  interview_matrices):
    """
    Aggiorna lo stato di sessione con:
    - obiettivo decisionale
    - lista criteri
    - lista alternative
    - matrice prestazioni alternative x criteri
    - tutte le matrici di confronto raccolte finora
    """
    st.session_state.backup_state["objective"] = objective
    st.session_state.backup_state["criteria"] = criteria_list
    st.session_state.backup_state["alternatives"] = alternatives_list

    perf_dict = {}
    for alt in alternatives_list:
        perf_dict[alt] = {}
        for crit in criteria_list:
            perf_dict[alt][crit] = float(perf_df.loc[alt, crit])
    st.session_state.backup_state["performance_matrix"] = perf_dict

    st.session_state.backup_state["num_interviews"] = num_interviews

    interviews_dict = {}
    for k, dfmat in interview_matrices.items():
        interviews_dict[str(k)] = {
            "pairwise": matrix_to_dict(dfmat)
        }
    st.session_state.backup_state["interviews"] = interviews_dict


def load_state_from_backup(backup_dict):
    """
    Ripristina i dati dal backup JSON.
    Restituisce tuple utili per popolazione dell'interfaccia.
    """
    try:
        objective = backup_dict["objective"]
        criteria_list = backup_dict["criteria"]
        alternatives_list = backup_dict["alternatives"]
        perf_mat_dict = backup_dict["performance_matrix"]
        num_interviews = backup_dict["num_interviews"]

        # ricostruisci performance DataFrame
        perf_df = pd.DataFrame.from_dict(perf_mat_dict, orient="index")
        perf_df = perf_df[criteria_list]  # garantisce l'ordine delle colonne

        # ricostruisci interviste
        interview_matrices = {}
        for key, sub in backup_dict["interviews"].items():
            interview_matrices[int(key)] = dict_to_matrix(sub["pairwise"])

        # salva anche nello stato sessione
        st.session_state.backup_state = backup_dict

        return (objective,
                criteria_list,
                alternatives_list,
                perf_df,
                num_interviews,
                interview_matrices)

    except Exception as e:
        st.error(f"Errore nel ripristino del backup: {e}")
        return None


# =========================================================
# FUNZIONI CORE AHP
# =========================================================

def create_empty_pairwise_matrix(elements):
    """
    Crea una matrice n x n con 1 sulla diagonale.
    """
    n = len(elements)
    mat = np.ones((n, n), dtype=float)
    return pd.DataFrame(mat, index=elements, columns=elements)


def saaty_scale_description():
    """
    Scala di Saaty 1..9 (descrizioni sintetiche).
    """
    return {
        1: "Uguale importanza",
        2: "Tra uguale e moderata",
        3: "Importanza moderata",
        4: "Tra moderata e forte",
        5: "Importanza forte",
        6: "Tra forte e molto forte",
        7: "Importanza molto forte",
        8: "Tra molto forte ed estrema",
        9: "Importanza estrema"
    }


def calculate_priority_vector(pairwise_matrix: pd.DataFrame):
    """
    Priority vector dei criteri:
    media delle righe dopo normalizzazione per colonna (metodo classico Saaty approx).
    """
    A = pairwise_matrix.values.astype(float)
    col_sum = A.sum(axis=0)
    norm = A / col_sum
    w = norm.mean(axis=1)
    return w  # numpy array di lunghezza n


def calculate_consistency_ratio(pairwise_matrix: pd.DataFrame, weights: np.ndarray):
    """
    Calcolo di λ_max, CI e CR per la matrice AHP.
    """
    A = pairwise_matrix.values.astype(float)
    n = A.shape[0]

    Aw = A @ weights
    lambda_max = np.mean(Aw / weights)

    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0

    RI_table = {
        1: 0.00,
        2: 0.00,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49
    }
    RI = RI_table.get(n, 1.49)
    if n <= 2:
        CR = 0.0
    else:
        CR = CI / RI if RI != 0 else 0.0

    return CR, CI, lambda_max


def geometric_mean(values):
    """
    Media geometrica di valori positivi.
    """
    arr = np.array(values, dtype=float)
    if np.any(arr <= 0):
        # in teoria non dovrebbe succedere con la scala Saaty positiva
        return 0.0
    prod = np.prod(arr)
    return prod ** (1.0 / len(arr))


def aggregate_experts_geometric_mean(list_of_matrices):
    """
    Aggregazione delle matrici di confronto dei vari esperti
    tramite media geometrica elemento-per-elemento.
    (Group AHP standard)
    """
    if len(list_of_matrices) == 1:
        return list_of_matrices[0].copy()

    idx = list_of_matrices[0].index
    cols = list_of_matrices[0].columns
    n = len(idx)

    agg = np.ones((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            vals = [m.iloc[i, j] for m in list_of_matrices]
            agg[i, j] = geometric_mean(vals)

    return pd.DataFrame(agg, index=idx, columns=cols)


# =========================================================
# FUNZIONI DI VISUALIZZAZIONE
# =========================================================

def plot_radar_chart(criteria_list, data_rows, labels):
    """
    Radar chart per confrontare alternative sui criteri normalizzati.
    """
    N = len(criteria_list)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

    for i, rowvals in enumerate(data_rows):
        vals = list(rowvals) + [rowvals[0]]
        ax.plot(angles, vals, linewidth=2, label=labels[i])
        ax.fill(angles, vals, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria_list, fontsize=9)
    ax.set_yticks([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.2,1.1), fontsize=9)

    st.pyplot(fig)


def plot_bar_ranking(df_ranked, alt_col_name, score_col_name):
    """
    Bar chart delle alternative ordinate per punteggio finale.
    """
    fig, ax = plt.subplots(figsize=(7,4))
    names = df_ranked[alt_col_name].tolist()
    scores = df_ranked[score_col_name].tolist()

    bars = ax.bar(names, scores)
    ax.set_ylabel("Score finale pesato")
    ax.set_title("Ranking alternative")
    ax.grid(axis='y', alpha=0.3)

    for b, s in zip(bars, scores):
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2., h, f"{s:.3f}",
                ha='center', va='bottom', fontsize=8)

    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig)


# =========================================================
# APP STREAMLIT
# =========================================================

def main():
    # impostazioni pagina (wide layout, favicon minimale)
    st.set_page_config(
        page_title="AHP Decision Tool",
        layout="wide",
        page_icon="📊"
    )

    # stato
    init_session_state()

    # HEADER COMPATTO
    title_col, info_col = st.columns([0.7, 0.3])
    with title_col:
        st.markdown("### 📊 AHP Decision Tool")
        st.caption("Valutazione multi-criterio con supporto multi-esperto")
    with info_col:
        st.caption("Versione prototipo • sessione locale")

    # TUTORIAL / ISTRUZIONI (collassabile)
    with st.expander("❓ Tutorial / Istruzioni per gli esperti"):
        st.markdown(TUTORIAL_PLACEHOLDER)

    st.divider()

    # =====================================================
    # STEP 1. OBIETTIVO DECISIONALE
    # =====================================================

    st.markdown("#### 1. Obiettivo")
    objective = st.text_input(
        "Obiettivo della decisione",
        value=st.session_state.backup_state.get("objective", ""),
        placeholder="Esempio: Selezionare il miglior DLM"
    )

    st.divider()

    # =====================================================
    # STEP 2. DATI DELLE ALTERNATIVE E CRITERI
    # =====================================================

    st.markdown("#### 2. Dati di input (Alternative e Criteri)")

    col_upload, col_backup = st.columns(2)

    with col_upload:
        st.write("Carica file con alternative e criteri (CSV o Excel).")
        st.caption(
            "- Prima colonna: nome alternativa (DLM)\n"
            "- Colonne successive: criteri numerici"
        )
        uploaded_data = st.file_uploader(
            "Carica dati",
            type=["csv", "xlsx", "xls"],
            key="uploaded_data_file"
        )

    with col_backup:
        st.write("Oppure riprendi un backup JSON completo.")
        uploaded_backup = st.file_uploader(
            "Carica backup JSON",
            type=["json"],
            key="uploaded_backup_file"
        )

    restored = None
    if uploaded_backup is not None:
        backup_dict = try_load_backup_json(uploaded_backup)
        if backup_dict is not None:
            restored = load_state_from_backup(backup_dict)
            st.success("Backup caricato. Interfaccia precompilata con i dati salvati.")

    # Se abbiamo ripristinato un backup
    if restored is not None:
        (objective_rest,
         criteria_list_rest,
         alternatives_list_rest,
         perf_df_rest,
         num_interviews_rest,
         interview_matrices_rest) = restored

        if not objective:
            objective = objective_rest
    else:
        criteria_list_rest = []
        alternatives_list_rest = []
        perf_df_rest = None
        num_interviews_rest = st.session_state.backup_state.get("num_interviews", 1)
        interview_matrices_rest = {}

    # Lettura file alternative/criteri se è stato caricato OR dal backup
    if uploaded_data is not None:
        df_raw = load_alternatives_file(uploaded_data)
        if df_raw is None or df_raw.empty:
            st.error("File vuoto/non valido.")
            st.stop()
    elif perf_df_rest is not None:
        df_raw = perf_df_rest.reset_index().rename(columns={"index": "Alternative"})
    else:
        df_raw = None

    if df_raw is not None:
        st.markdown("**Anteprima dati caricati**")
        st.dataframe(df_raw.head(), use_container_width=True)

        alt_col_name = df_raw.columns[0]
        alternatives_list = df_raw[alt_col_name].astype(str).tolist()
        criteria_list = list(df_raw.columns[1:])
        perf_df = df_raw.set_index(alt_col_name).astype(float)

        st.caption(f"Alternative rilevate: {alternatives_list}")
        st.caption(f"Criteri rilevati: {criteria_list}")
    else:
        alternatives_list = alternatives_list_rest
        criteria_list = criteria_list_rest
        perf_df = perf_df_rest if perf_df_rest is not None else pd.DataFrame()

    if len(criteria_list) == 0 or len(alternatives_list) == 0:
        st.warning("Carica un file dati o un backup per continuare con i passi successivi.")
        return  # fermiamo qui la UI successiva

    st.divider()

    # =====================================================
    # STEP 3. NUMERO DI INTERVISTE (ESPERTI)
    # =====================================================

    st.markdown("#### 3. Esperti / Interviste")
    num_interviews = st.number_input(
        "Numero di esperti/interviste",
        min_value=1,
        max_value=20,
        step=1,
        value=num_interviews_rest,
        help="Per ogni esperto raccoglieremo una matrice di confronto tra i criteri."
    )

    st.caption(
        "Ogni esperto compila confronti a coppie tra i criteri "
        "usando la scala 1–9 di Saaty. "
        "I giudizi degli esperti saranno poi aggregati tramite media geometrica."
    )

    st.divider()

    # =====================================================
    # STEP 4. CONFRONTI A COPPIE (SCALA SAATY) PER OGNI ESPERTO
    # =====================================================

    st.markdown("#### 4. Confronti tra criteri (pairwise)")

    saaty_desc = saaty_scale_description()
    st.caption(
        "Scala Saaty: 1 = uguale importanza, 9 = importanza estremamente maggiore."
    )

    # Struttura per salvare le matrici di confronto
    interview_matrices = {}
    if len(interview_matrices_rest) > 0:
        # se abbiamo già caricato dal backup, partiamo da lì
        for k, dfmat in interview_matrices_rest.items():
            interview_matrices[k] = dfmat.copy()

    n_crit = len(criteria_list)

    for interview_id in range(num_interviews):
        st.markdown(f"**Esperto #{interview_id + 1}**")

        # se abbiamo una matrice già salvata la riutilizziamo, altrimenti creiamo matrice identità
        if interview_id in interview_matrices:
            pairwise_df = interview_matrices[interview_id]
        else:
            pairwise_df = create_empty_pairwise_matrix(criteria_list)

        with st.expander(f"Confronti criteri – Esperto #{interview_id + 1}", expanded=True):
            for i in range(n_crit):
                for j in range(i+1, n_crit):
                    left = criteria_list[i]
                    right = criteria_list[j]

                    st.write(f"{left} ↔ {right}")

                    pref = st.radio(
                        "Quale criterio è più importante?",
                        options=[left, "Uguali", right],
                        index=1,
                        key=f"pref-{interview_id}-{i}-{j}"
                    )

                    # slider intensità
                    default_val = pairwise_df.loc[left, right]
                    if default_val < 1:
                        # se è reciprocale <1 non posso metterlo come default dello slider,
                        # lo riporto a 1 perché lo slider va solo da 1 a 9
                        default_slider_val = 1
                    else:
                        default_slider_val = int(round(default_val))

                    intensity = st.slider(
                        "Intensità importanza (1 = uguale, 9 = estremamente più importante)",
                        min_value=1,
                        max_value=9,
                        value=default_slider_val,
                        step=1,
                        key=f"intensity-{interview_id}-{i}-{j}"
                    )

                    # logica AHP standard (reciprocità)
                    if pref == "Uguali" or intensity == 1:
                        val = 1.0
                    elif pref == left:
                        val = float(intensity)
                    else:  # pref == right
                        val = 1.0 / float(intensity)

                    pairwise_df.loc[left, right] = val
                    pairwise_df.loc[right, left] = 1.0 / val
                    pairwise_df.loc[left, left] = 1.0
                    pairwise_df.loc[right, right] = 1.0

            st.caption("Matrice di confronto (criteri) per questo esperto")
            st.dataframe(pairwise_df.style.format("{:.4f}"), use_container_width=True)

        # aggiorno dizionario
        interview_matrices[interview_id] = pairwise_df

    st.divider()

    # =====================================================
    # STEP 5. BACKUP STATO
    # =====================================================

    st.markdown("#### 5. Backup stato (facoltativo)")
    st.caption(
        "Salva lo stato corrente (inclusi confronti degli esperti) in un JSON. "
        "Questo permette di interrompere e riprendere più tardi replicando l'esperimento."
    )

    save_current_state_to_session(
        objective,
        criteria_list,
        alternatives_list,
        perf_df,
        num_interviews,
        interview_matrices
    )

    download_json_button(
        "💾 Scarica backup (.json)",
        st.session_state.backup_state,
        "AHP_backup.json"
    )

    st.divider()

    # =====================================================
    # STEP 6. CALCOLO FINALE AHP
    # =====================================================

    st.markdown("#### 6. Calcolo finale e ranking")
    st.caption("Calcola pesi dei criteri (AHP), consistenza e punteggi finali delle alternative.")

    if st.button("Esegui calcolo AHP", type="primary"):
        # 1. aggrego le matrici degli esperti con media geometrica
        matrices_list = [interview_matrices[i] for i in range(num_interviews)]
        final_criteria_matrix = aggregate_experts_geometric_mean(matrices_list)

        st.markdown("**Matrice criteri aggregata (media geometrica tra esperti)**")
        st.dataframe(
            final_criteria_matrix.style.format("{:.4f}"),
            use_container_width=True
        )

        # 2. priority vector dei criteri
        w = calculate_priority_vector(final_criteria_matrix)
        weights_criteria = pd.Series(w, index=criteria_list, name="Weight")

        st.markdown("**Pesi dei criteri (priority vector)**")
        st.dataframe(
            weights_criteria.to_frame(),
            use_container_width=True
        )

        # 3. Consistency Ratio
        CR, CI, lambda_max = calculate_consistency_ratio(final_criteria_matrix, w)

        st.markdown("**Consistenza (Saaty)**")
        col_ci, col_cr, col_lambda = st.columns(3)
        with col_lambda:
            st.metric("λ_max", f"{lambda_max:.4f}")
        with col_ci:
            st.metric("CI", f"{CI:.4f}")
        with col_cr:
            st.metric("CR", f"{CR:.4f}")

        if CR > 0.10:
            st.warning("CR > 0.10 → giudizi poco consistenti, considerare revisione.")
        else:
            st.success("CR ≤ 0.10 → giudizi coerenti.")

        # 4. calcolo punteggi finali alternative = somma pesata prestazioni
        scores = {}
        for alt in alternatives_list:
            row_vals = perf_df.loc[alt, criteria_list].values.astype(float)
            scores[alt] = np.dot(row_vals, weights_criteria.values)

        scores_series = pd.Series(scores, name="FinalScore")

        result_df = perf_df.copy()
        result_df["FinalScore"] = scores_series
        result_ranked = result_df.sort_values(by="FinalScore", ascending=False).reset_index()
        result_ranked.rename(columns={"index": "Alternative"}, inplace=True)
        result_ranked["Rank"] = np.arange(1, len(result_ranked)+1)

        st.markdown("**Ranking finale delle alternative**")
        st.dataframe(
            result_ranked[["Rank","Alternative","FinalScore"] + criteria_list] \
                .style.format(
                    {"FinalScore":"{:.4f}", **{c:"{:.2f}" for c in criteria_list}}
                ) \
                .background_gradient(subset=["FinalScore"], cmap="viridis"),
            use_container_width=True
        )

        best_alt = result_ranked.iloc[0]["Alternative"]
        best_score = result_ranked.iloc[0]["FinalScore"]
        st.success(f"Alternativa migliore: {best_alt} (score {best_score:.4f})")

        st.divider()

        # 5. Visualizzazioni compatte
        st.markdown("#### Visualizzazioni")

        tab_bar, tab_radar, tab_weights = st.tabs(["Ranking", "Radar", "Pesi criteri"])

        with tab_bar:
            plot_bar_ranking(result_ranked, "Alternative", "FinalScore")

        with tab_radar:
            # normalizzazione 0-1 per ogni criterio per lettura comparativa
            radar_df = perf_df.copy()
            for c in criteria_list:
                col = radar_df[c].astype(float)
                maxv = col.max()
                radar_df[c] = col / maxv if maxv > 0 else 0.0

            radar_rows = [
                radar_df.loc[alt, criteria_list].values.astype(float)
                for alt in alternatives_list
            ]
            plot_radar_chart(criteria_list, radar_rows, alternatives_list)

        with tab_weights:
            fig, ax = plt.subplots(figsize=(5,3))
            bars = ax.bar(criteria_list, weights_criteria.values)
            ax.set_ylabel("Peso")
            ax.set_title("Pesi dei criteri")
            ax.grid(axis='y', alpha=0.3)
            for b, s in zip(bars, weights_criteria.values):
                h = b.get_height()
                ax.text(
                    b.get_x()+b.get_width()/2., h,
                    f"{s:.3f}",
                    ha='center', va='bottom', fontsize=8
                )
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

        st.divider()

        # 6. Export risultati
        st.markdown("#### Esporta risultati")

        output_excel = io.BytesIO()
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            result_ranked.to_excel(writer, sheet_name='Ranking', index=False)
            weights_criteria.to_frame().to_excel(writer, sheet_name='CriteriaWeights')
            final_criteria_matrix.to_excel(writer, sheet_name='CriteriaMatrix')
        output_excel.seek(0)

        st.download_button(
            label="📥 Scarica risultati (Excel)",
            data=output_excel,
            file_name="AHP_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        csv_bytes = result_ranked.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Scarica ranking (CSV)",
            data=csv_bytes,
            file_name="AHP_ranking.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()