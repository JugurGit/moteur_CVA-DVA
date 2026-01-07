# app_streamlit.py
from __future__ import annotations

import streamlit as st

from app_lib.style import apply_page_config, apply_css

# ---------------------------------------------------------------------
# 0) Setup UI (config + CSS)
# ---------------------------------------------------------------------
# On centralise ici le thème, le layout, les styles "pro", etc.
apply_page_config(title="Réplication du moteur CVA/DVA", icon="📊")
apply_css()

# ---------------------------------------------------------------------
# 1) Sidebar globale (comme dans app.py)
# ---------------------------------------------------------------------
# Objectif : donner un point d’entrée clair (contexte + mode d’usage),
# et stocker quelques toggles dans session_state pour les autres pages.
with st.sidebar:
    st.markdown("## XVA Lab")
    st.caption("CVA/DVA • Hull–White 1F++ • log-OU • Shapley • Tracking")

    # Mode "tracking" : utile si tu historises des runs / snapshots dans l'app
    tracking = st.toggle("📌 Portfolio tracking mode", value=True)
    st.session_state["tracking_mode"] = tracking

    # Optionnel : un mode verbose global (pratique si tu veux afficher plus de logs)
    verbose = st.toggle("🧪 Mode verbose", value=False)
    st.session_state["verbose_mode"] = verbose

    st.divider()

# ---------------------------------------------------------------------
# 2) Page content (marketing / mémoire) — structuré comme app.py
# ---------------------------------------------------------------------
st.title("📊 Réplication du moteur CVA/DVA — Démo technique")
st.caption("Scénarios taux & crédit • Expositions EPE/ENE • CVA/DVA • Explain (Shapley) • Export & traçabilité")

st.markdown("### 🧩 Contexte — De Banque Palatine à une démo “reproductible”")

st.info(
    """
Lors de mon stage de fin d’études chez Banque Palatine (Département RISF),
 j’ai travaillé sur le calcul des métriques de risque de contrepartie, CVA et DVA, ainsi que sur leurs sensibilités.
Le but était de réconcilier ses chiffres avec ceux du moteur AmerisC (moteur de calcul de risque de contrepartie de Natixis).
 Ce projet est donc une **démo technique** inspirée des travaux réalisés chez **Banque Palatine**.
L’objectif est de **recréer une chaîne de calcul CVA/DVA** dans un cadre **structuré comme en production** :

- **Génération de scénarios** (taux) via **Hull–White 1F++**  
- **Modélisation des intensités de défaut** via **log-OU** (contreparties **et** banque)  
- **Expositions** (**EPE / ENE**) puis calcul des **legs** et **totaux** de **CVA / DVA**  
- **Traçabilité** : logs, snapshots, exports (CSV / JSON / PNG) pour reproduire et documenter un run
""",
    icon="🏦",
)

st.warning(
    """
Je ne dispose pas des **données internes** ni de la **documentation** nécessaires
pour illustrer les traitements de manière “réelle”.
Le projet remplace donc ces entrées par des données **contrôlées / simulées**.
""",
    icon="⚠️",
)

st.markdown("### 🎯 Ce que démontre ce mini-projet (workflow end-to-end)")

cA, cB, cC, cD = st.columns(4)
with cA:
    st.markdown("**1) Hypothèses maîtrisées**")
    st.caption("Marché simulé • seeds • horizons • paramètres modèles")
with cB:
    st.markdown("**2) Simulation & expositions**")
    st.caption("Trajectoires • cashflows • EPE/ENE • profils temporels")
with cC:
    st.markdown("**3) CVA/DVA calculés**")
    st.caption("Discounting • PD • agrégation buckets • résultats exploitables")
with cD:
    st.markdown("**4) Explicabilité & traçabilité**")
    st.caption("Shapley • contributions • exports • comparaisons de runs")

st.success(
    """
**En résumé** : une réplique “mini moteur” qui illustre **la même démarche que chez Banque Palatine** :
structurer un calcul XVA avec des inputs maîtrisés, des sorties traçables, et une lecture claire des **sensibilités**
(exposition, discounting, probabilités de défaut).
""",
    icon="✅",
)


# ---------------------------------------------------------------------
# 3) Navigation (comme ton app.py FRTB / IR Lab)
# ---------------------------------------------------------------------
st.markdown(
    """
### 🧭 Navigation
Utilisez les pages à gauche :

- **Overview** : résumé + état courant + KPIs (CVA, DVA, EPE, ENE)
- **Market / Models** : hypothèses simulées (HW 1F++ / log-OU), paramètres, seeds
- **Run / Simulation** : exécution d’un run, suivi logs, sauvegarde des artefacts
- **Exposures** : profils EPE/ENE (agrégé / par contrepartie)
- **CVA / DVA** : legs (DF, PD, expo) + totaux + vues par buckets
- **Analytics** : Shapley / contributions (DF, expo, PD) par bucket et/ou contrepartie
- **Export** : CSV / JSON / PNG pour reporting et historique

> Astuce : si le moteur imprime beaucoup, on capture les logs et on les affiche pour garder une trace du run.
"""
)

# ---------------------------------------------------------------------
# 4) (Optionnel) Affichage des logs du dernier run, si disponibles
# ---------------------------------------------------------------------
# Si tes pages "Run" stockent des logs dans session_state, ce bloc les rend accessibles depuis l'accueil.
if st.session_state.get("last_logs"):
    with st.expander("Afficher les logs du dernier run", expanded=False):
        st.code(st.session_state["last_logs"], language="text")
