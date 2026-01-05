# app_streamlit.py
from __future__ import annotations

import streamlit as st

from app_lib.style import apply_page_config, apply_css

apply_page_config(title="Réplication du moteur CVA/DVA", icon="📊")
apply_css()

st.title("📊 Réplication du moteur CVA/DVA")

st.markdown(
    """
Ce mini-projet **reproduit, à des fins pédagogiques, un moteur de calcul CVA/DVA** inspiré des travaux réalisés chez **Banque Palatine**.

Faute de données de marché complètes, les entrées nécessaires sont **simulées** :
- **Scénarios de taux** via un modèle **Hull–White 1F++**
- **Intensités de défaut** via un modèle **log-OU** (contreparties **et** banque)

Le moteur calcule ensuite :
- les **expositions** (**EPE / ENE**),
- les **legs** et **totaux** de **CVA / DVA**, avec **agrégation par buckets**,
- l’**export** des résultats (**CSV / JSON / PNG**).

Enfin, l’application propose des analyses complémentaires :
- **décomposition de type Shapley** des contributions (**DF**, **expositions**, **probabilités de défaut**).
"""
)


st.markdown(
    """
### 🧭 Parcours conseillé (3–5 minutes)

1. **Overview**
   - Vérifier l’**état du run courant** (date, modèle, taille de simulation).
   - Repèrer les **KPIs clés** (CVA, DVA, EPE, ENE) pour avoir un point de départ.

2. **Market / Models**
   - Consulter les **hypothèses de marché simulées** :
     - courbes / paramètres **Hull–White 1F++**
     - intensités **log-OU** (contreparties + banque)
   - Ajuster si besoin les paramètres (vol, mean reversion, seeds, horizons).

3. **Run / Simulation**
   - Lancer un **run complet** (ou recharger un run existant si l’app le permet).
   - Surveiller les logs/infos de calcul et valider que l’export est généré.

4. **Exposures**
   - Visualiser les profils **EPE / ENE** (par contrepartie et/ou agrégé).
   - Identifier rapidement les **drivers** (maturité, notionnel, sens payer/receiver).

5. **CVA / DVA**
   - Examiner les **legs** (discounting, PD, exposition) puis les **totaux**.
   - Passer en vue **bucket** pour comprendre l’agrégation et les contributions.

6. **Analytics**
   - **Shapley** : décomposer les contributions (DF / exposition / PD) par bucket.

7. **Export**
   - Récupèrer les résultats (CSV/JSON/PNG) pour garder une trace ou alimenter un reporting.
"""
)

