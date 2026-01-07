# Moteur CVA / DVA — Mini Dashboard (Streamlit)

Dashboard Streamlit multi-pages qui expose une **démo pédagogique** d’un moteur **CVA / DVA** (type portefeuille IRS) :
- **Taux** : scénarios simulés via **Hull–White 1F++** calibré pour recoller une courbe initiale (**Nelson–Siegel**)
- **Crédit** : intensités simulées via **Log-OU** (banque + contreparties), avec **Survie** et **PD marginales**
- **Exposition** : **MTM par scénario** → profils **EPE / ENE**
- **xVA** : calcul des **legs bucketés** + agrégation **CVA / DVA**
- **Explicabilité** : décomposition **Shapley** (contributions DF / EPE(ENE) / PD / Survie)
- **Traçabilité** : exports CSV/JSON/PNG, runs persistés dans `./data/`

> ⚠️ Projet à but illustratif : les modèles, paramètres et données “demo” sont simplifiés et ne constituent pas un moteur de production.

👉 Démo en ligne : **https://boudarene-moteurcvadva.streamlit.app/**
---

## 1) Prérequis

- **Python 3.10+** (recommandé)

---

### 2) Récupérer le projet
#### Option A — via Git
```bash
git clone <URL_DU_REPO>
cd <NOM_DU_REPO>
```

#### Option B — via ZIP
- Télécharger le ZIP depuis GitHub
- Le dézippez
- Ouvrir un terminal dans le dossier du projet

### 3) Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4) Lancer l’application Streamlit
```bash
streamlit run app_streamlit.py
```
Streamlit va afficher une URL du type :
- Local: http://localhost:8501

### 5) Utilisation rapide

#### Ouvrir Simulation : configurer
- Nombre de contreparties
- Nombre de scénarios N
- Seed (reproductibilité)
- Options : exports PNG, Snapshot Mar + compare + Shapley
#### Lancer Run
#### Ouvrir Dashboard : consulter CVA/DVA totaux, legs agrégés, téléchargements rapides
#### Ouvrir Contreparties : drilldown EPE/ENE, PD/Survie, legs CVA/DVA par contrepartie
#### Ouvrir Sensibilités : Shapley CVA/DVA + (optionnel) comparatif Jan/Mar (PV Jan)
#### Ouvrir Portfolio Tracking : ranking (CVA/DVA/Net), compare run-vs-run, exports CSV
#### Ouvrir Documentation : fiches manuelles (pilotées par pages/docs_registry.json)


### 6) Lancer le moteur en ligne de commande via un notebook

Le script main.py exécute un run “console” :

**Mode démo**
```bash
python test.ipynb
```
