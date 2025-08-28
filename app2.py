import streamlit as st
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64
import requests
import os

# Configuration de la page
st.set_page_config(
    page_title="DILI Predictor - Toxicité Hépatique",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration GitHub Releases
GITHUB_REPO = "kym94/DILI-Prediction-App"
MODEL_VERSION = "v1.0"
MODEL_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{MODEL_VERSION}/best_dili_model_20250826_002227.pkl"


@st.cache_resource
def download_model():
    """Télécharge le modèle si nécessaire"""
    model_filename = "best_dili_model_20250826_002227.pkl"

    if not os.path.exists(model_filename):
        with st.spinner("Premier lancement : téléchargement du modèle (49.5 MB)..."):
            try:
                response = requests.get(MODEL_URL, timeout=300)  # 5 min timeout
                response.raise_for_status()

                with open(model_filename, 'wb') as f:
                    f.write(response.content)

                st.success("✅ Modèle téléchargé avec succès!")
                return True
            except Exception as e:
                st.error(f"❌ Erreur téléchargement modèle: {e}")
                return False
    return True


# Chargement du modèle (mise en cache)
@st.cache_resource
def load_model():
    # S'assurer que le modèle est téléchargé
    if not download_model():
        st.error("Impossible de télécharger le modèle")
        return None

    try:
        return joblib.load('best_dili_model_20250826_002227.pkl')
    except FileNotFoundError:
        st.error("Modèle non trouvé après téléchargement.")
        return None
    except Exception as e:
        st.error(f"Erreur chargement modèle: {e}")
        return None


# Chargement de la chimiothèque de pharmacopée (mise en cache)
@st.cache_data
def load_pharmacopee_database():
    try:
        df = pd.read_excel('molecules pharmacopee OOAS et proprites.xlsx')

        # Corriger la structure des données : remplir les cellules vides de 'Plante'
        # avec la dernière plante mentionnée (forward fill)
        df['Plante'] = df['Plante'].fillna(method='ffill')

        # Nettoyage des données
        df = df.dropna(subset=['Canonical SMILES'])  # Supprimer lignes sans SMILES
        df = df.reset_index(drop=True)

        return df
    except FileNotFoundError:
        st.error(
            "Chimiothèque pharmacopée non trouvée. Assurez-vous que 'molecules pharmacopee OOAS et proprites.xlsx' est dans le répertoire.")
        return None


def create_pharmacopee_visualizations(df):
    """Créer des visualisations pour la base de données pharmacopée"""
    # Distribution des poids moléculaires
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Distribution MW
    axes[0, 0].hist(df['MW'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution des Poids Moléculaires')
    axes[0, 0].set_xlabel('MW (g/mol)')
    axes[0, 0].set_ylabel('Fréquence')
    axes[0, 0].axvline(500, color='red', linestyle='--', label='Limite Lipinski (500)')
    axes[0, 0].legend()

    # 2. LogP vs TPSA
    scatter = axes[0, 1].scatter(df['Consensus Log P'].dropna(), df['TPSA'].dropna(),
                                 alpha=0.6, c=df['MW'].dropna(), cmap='viridis')
    axes[0, 1].set_title('LogP vs TPSA (coloré par MW)')
    axes[0, 1].set_xlabel('Consensus Log P')
    axes[0, 1].set_ylabel('TPSA (Å²)')
    plt.colorbar(scatter, ax=axes[0, 1], label='MW')

    # 3. Violations Lipinski
    lipinski_counts = df['Lipinski #violations'].value_counts().sort_index()
    axes[1, 0].bar(lipinski_counts.index, lipinski_counts.values, alpha=0.7, color='lightcoral')
    axes[1, 0].set_title('Violations de la Règle de Lipinski')
    axes[1, 0].set_xlabel('Nombre de violations')
    axes[1, 0].set_ylabel('Nombre de molécules')

    # 4. Absorption GI
    gi_counts = df['GI absorption'].value_counts()
    colors = ['lightgreen' if x == 'High' else 'orange' for x in gi_counts.index]
    axes[1, 1].pie(gi_counts.values, labels=gi_counts.index, autopct='%1.1f%%', colors=colors)
    axes[1, 1].set_title('Absorption Gastro-Intestinale')

    plt.tight_layout()
    return fig


def filter_pharmacopee_data(df, filters):
    """Filtrer la base de données selon les critères"""
    filtered_df = df.copy()

    if filters['mw_range'][0] < filters['mw_range'][1]:
        filtered_df = filtered_df[
            (filtered_df['MW'] >= filters['mw_range'][0]) &
            (filtered_df['MW'] <= filters['mw_range'][1])
            ]

    if filters['logp_range'][0] < filters['logp_range'][1]:
        filtered_df = filtered_df[
            (filtered_df['Consensus Log P'] >= filters['logp_range'][0]) &
            (filtered_df['Consensus Log P'] <= filters['logp_range'][1])
            ]

    if filters['gi_absorption'] != 'Tous':
        filtered_df = filtered_df[filtered_df['GI absorption'] == filters['gi_absorption']]

    if filters['lipinski_violations'] is not None:
        filtered_df = filtered_df[filtered_df['Lipinski #violations'] <= filters['lipinski_violations']]

    if filters['plante_search']:
        filtered_df = filtered_df[
            filtered_df['Plante'].str.contains(filters['plante_search'], case=False, na=False)
        ]

    if filters['molecule_search']:
        filtered_df = filtered_df[
            filtered_df['Molecule'].str.contains(filters['molecule_search'], case=False, na=False)
        ]

    return filtered_df


# Liste exacte des 36 descripteurs sélectionnés par ElasticNet
SELECTED_DESCRIPTORS = [
    'SlogP_VSA8', 'NHOHCount', 'H-Bond_Donor_Count', 'fr_Ar_NH', 'BCUT2D_MRHI',
    'SMR_VSA6', 'PEOE_VSA10', 'SlogP_VSA10', 'Total_Atom_Stereo_Count', 'SMR_VSA4',
    'fr_piperdine', 'NumRadicalElectrons', 'fr_tetrazole', 'fr_nitro', 'PEOE_VSA9',
    'XLogP', 'Charge', 'fr_quatN', 'BCUT2D_MRLOW', 'fr_Ndealkylation2',
    'fr_Al_OH', 'fr_pyridine', 'fr_hdrzine', 'fr_HOCCN', 'qed',
    'EState_VSA3', 'NumUnspecifiedAtomStereoCenters', 'fr_Ndealkylation1', 'PEOE_VSA8',
    'fr_furan', 'PEOE_VSA3', 'FractionCSP3', 'VSA_EState5', 'fr_guanido',
    'fr_aniline', 'Kappa1'
]


def calculate_descriptors(smiles):
    """Calcule les descripteurs EXACTS utilisés pour l'entraînement du modèle DILI"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Calculer automatiquement TOUS les descripteurs RDKit disponibles
        descriptors = {}

        # Parcourir tous les descripteurs disponibles dans RDKit
        for desc_name, desc_func in Descriptors.descList:
            try:
                descriptors[desc_name] = desc_func(mol)
            except:
                descriptors[desc_name] = 0.0

        # Ajouter quelques descripteurs personnalisés si nécessaire
        try:
            # Charge formelle totale de la molécule
            descriptors['Charge'] = sum([atom.GetFormalCharge() for atom in mol.GetAtoms()])

            # Calculer le nombre total d'atomes stéréogéniques (incluant assignés et non-assignés)
            chiral_centers_all = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            descriptors['Total_Atom_Stereo_Count'] = len(chiral_centers_all)

            # Nombre d'atomes avec stéréochimie non spécifiée
            chiral_centers_assigned = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
            descriptors['NumUnspecifiedAtomStereoCenters'] = len(chiral_centers_all) - len(chiral_centers_assigned)

            # Renommer quelques descripteurs pour correspondre aux noms utilisés
            if 'NumHDonors' in descriptors:
                descriptors['H-Bond_Donor_Count'] = descriptors['NumHDonors']

        except Exception as e:
            print(f"Erreur descripteurs personnalisés: {e}")
            # Valeurs par défaut en cas d'erreur
            descriptors['Total_Atom_Stereo_Count'] = 0
            descriptors['NumUnspecifiedAtomStereoCenters'] = 0

        # Sélectionner uniquement les 36 descripteurs utilisés par le modèle
        selected_desc = {}
        missing_descriptors = []

        for desc_name in SELECTED_DESCRIPTORS:
            if desc_name in descriptors:
                selected_desc[desc_name] = descriptors[desc_name]
            else:
                # Chercher des noms alternatifs courants
                found = False
                alternative_names = {
                    'H-Bond_Donor_Count': ['NumHDonors', 'HBD'],
                    'Total_Atom_Stereo_Count': ['NumAtomStereoCenters'],
                    'XLogP': ['MolLogP'],
                }

                if desc_name in alternative_names:
                    for alt_name in alternative_names[desc_name]:
                        if alt_name in descriptors:
                            selected_desc[desc_name] = descriptors[alt_name]
                            found = True
                            break

                if not found:
                    missing_descriptors.append(desc_name)
                    selected_desc[desc_name] = 0.0  # Valeur par défaut

        if missing_descriptors and len(missing_descriptors) < 5:  # Ne pas spammer si trop d'erreurs
            st.warning(f"Descripteurs manquants (remplacés par 0): {', '.join(missing_descriptors[:5])}")

        # Calculer les 1024 Morgan R2 fingerprints avec les noms exacts
        from rdkit.Chem import rdMolDescriptors
        morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        for i in range(1024):
            bit_name = f"Morgan_R2_1024_bit_{i:04d}"
            selected_desc[bit_name] = int(morgan_fp[i])

        return selected_desc

    except Exception as e:
        st.error(f"Erreur calcul descripteurs: {e}")
        return None


def predict_dili(model, features_dict):
    """Fait la prédiction DILI"""
    try:
        # Convertir en DataFrame avec ordre des colonnes attendu par le modèle
        df = pd.DataFrame([features_dict])

        # Vérifier que toutes les features attendues sont présentes
        expected_features = 36 + 1024  # 36 descripteurs + 1024 Morgan
        if len(df.columns) != expected_features:
            st.warning(f"Nombre de features: {len(df.columns)}, attendu: {expected_features}")

        # Prédiction
        prediction = model.predict(df)[0]

        # Probabilités si disponibles
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)[0]
            confidence = max(probabilities)
        else:
            # Fallback avec decision_function pour SVM
            if hasattr(model, 'decision_function'):
                scores = model.decision_function(df)
                probabilities = [0.5 - scores[0] / 10, 0.5 + scores[0] / 10]  # Approximation
                probabilities = np.clip(probabilities, 0, 1)
            else:
                probabilities = [0.5, 0.5]
            confidence = max(probabilities)

        return prediction, probabilities, confidence

    except Exception as e:
        st.error(f"Erreur prédiction: {e}")
        return None, None, None


def draw_molecule(smiles, size=(300, 200)):
    """Dessine la structure 2D de la molécule"""
    try:
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size)
        return img
    except ImportError:
        st.info("Visualisation moléculaire non disponible sur cette plateforme")
        return None
    except Exception:
        return None


def calculate_applicability_domain(features_dict, reference_ranges=None):
    """Calcule si la molécule est dans le domaine d'applicabilité du modèle"""
    # Simplifié - en production, utiliser les statistiques du training set
    warnings = []

    # Vérifications basiques
    if features_dict.get('XLogP', 0) > 10 or features_dict.get('XLogP', 0) < -5:
        warnings.append("LogP extrême (hors gamme typique)")

    if features_dict.get('H-Bond_Donor_Count', 0) > 20:
        warnings.append("Trop de donneurs de liaison H")

    return warnings


# CSS personnalisé
def local_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-positive {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .prediction-negative {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


# Interface principale
def main():
    local_css()

    # Header
    st.markdown("<h1 class='main-header'>🧬 TOXICITE HEPATIQUE - PHARMACOPÉE OUEST-AFRICAINE - MACHINE LEARNING</h1>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1.2rem; color: #666;'>Prédiction de toxicité hépatique médicamenteuse par intelligence artificielle</p>",
        unsafe_allow_html=True)

    # Sidebar avec informations modèle
    with st.sidebar:
        st.header("📊 Informations Modèle")
        st.markdown("""
        **Modèle principal:** Stacking Classifier  
        **Modèles de base:**
        • Random Forest
        • Support Vector Machine (linéaire & RBF)  
        • k-Nearest Neighbors

        **Meta-estimateur:** Logistic Regression

        **Features:** 36 descripteurs + 1024 Morgan R2  
        **F1-score externe:** 75.6%  
        **AUC externe:** 79.5%  
        **Dataset d'entraînement:** 945 molécules  
        **Validation externe:** 156 molécules
        """)

        st.warning("⚠️ **Outil de recherche uniquement**\nPas à usage clinique")

        # Afficher des exemples de SMILES
        st.subheader("🧪 Exemples SMILES")
        examples = {
            "Éthanol (Non-DILI)": "CCO",
            "Paracétamol (DILI)": "CC(=O)Nc1ccc(O)cc1",
            "Aspirine (DILI)": "CC(=O)Oc1ccccc1C(=O)O",
            "Caféine (Non-DILI)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        }

        for name, smiles in examples.items():
            if st.button(f"Charger {name.split('(')[0].strip()}", key=f"example_{name}"):
                st.session_state.example_smiles = smiles

    # Interface principale
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Prédiction Unique", "📊 Analyse Batch", "🧪 Chimiothèque", "📖 À propos"])

    with tab1:
        st.header("Prédiction pour une molécule")

        # Input SMILES
        example_smiles = st.session_state.get('example_smiles', '')
        smiles_input = st.text_input(
            "Entrez le code SMILES:",
            value=example_smiles,
            placeholder="Exemple: CCO (éthanol)",
            help="Le SMILES (Simplified Molecular Input Line Entry System) représente la structure chimique"
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            predict_btn = st.button("🚀 Prédire", type="primary")

        with col2:
            if smiles_input:
                # Afficher la structure moléculaire
                mol_img = draw_molecule(smiles_input, size=(250, 200))
                if mol_img:
                    st.image(mol_img, caption="Structure 2D", width=250)

        if predict_btn and smiles_input:
            with st.spinner("Calcul des descripteurs et prédiction en cours..."):
                # Charger le modèle
                model = load_model()
                if model is None:
                    return

                # Calculer descripteurs
                features = calculate_descriptors(smiles_input)

                if features:
                    # Prédiction
                    pred, probs, conf = predict_dili(model, features)

                    if pred is not None:
                        # Résultats principaux
                        if pred == 1:
                            st.markdown("""
                            <div class='prediction-positive'>
                            <h3>🔴 Prédiction: DILI POSITIF</h3>
                            <p>La molécule est prédite comme potentiellement hépatotoxique</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class='prediction-negative'>
                            <h3>🟢 Prédiction: Non-DILI</h3>
                            <p>La molécule est prédite comme non hépatotoxique</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Métriques détaillées
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            result_text = "DILI+" if pred == 1 else "DILI-"
                            st.metric("Classe", result_text)

                        with col2:
                            if probs is not None and len(probs) > 1:
                                dili_prob = probs[1] * 100
                                st.metric("Probabilité DILI", f"{dili_prob:.1f}%")

                        with col3:
                            st.metric("Confiance", f"{conf * 100:.1f}%")

                        with col4:
                            risk_level = "Élevé" if probs[1] > 0.7 else "Modéré" if probs[1] > 0.3 else "Faible"
                            st.metric("Niveau risque", risk_level)

                        # Graphique des probabilités
                        if probs is not None and len(probs) > 1:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                            # Graphique en barres
                            labels = ['Non-DILI', 'DILI']
                            colors = ['#4caf50', '#f44336']
                            bars = ax1.bar(labels, probs, color=colors, alpha=0.7, edgecolor='black')
                            ax1.set_ylabel('Probabilité')
                            ax1.set_title('Distribution des Probabilités')
                            ax1.set_ylim(0, 1)

                            # Ajouter les valeurs sur les barres
                            for bar, prob in zip(bars, probs):
                                height = bar.get_height()
                                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                         f'{prob:.3f}', ha='center', va='bottom')

                            # Graphique en secteurs
                            ax2.pie(probs, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                            ax2.set_title('Répartition des Probabilités')

                            plt.tight_layout()
                            st.pyplot(fig)

                        # Vérification domaine d'applicabilité
                        warnings = calculate_applicability_domain(features)
                        if warnings:
                            st.markdown("""
                            <div class='warning-box'>
                            <h4>⚠️ Avertissements sur le domaine d'applicabilité:</h4>
                            """, unsafe_allow_html=True)
                            for warning in warnings:
                                st.write(f"• {warning}")
                            st.markdown("</div>", unsafe_allow_html=True)

                        # Avertissements sur la confiance
                        if conf < 0.6:
                            st.error("⚠️ Prédiction peu fiable (confiance < 60%). Interpréter avec prudence.")
                        elif conf < 0.7:
                            st.warning("⚠️ Prédiction modérément fiable (confiance < 70%).")

                        # Détails des descripteurs (expandeur)
                        with st.expander("📊 Détails des descripteurs calculés"):
                            desc_df = pd.DataFrame([
                                {"Descripteur": k, "Valeur": v}
                                for k, v in features.items()
                                if k in SELECTED_DESCRIPTORS
                            ])
                            st.dataframe(desc_df, use_container_width=True)

                    else:
                        st.error("Erreur lors de la prédiction")
                else:
                    st.error("SMILES invalide ou erreur de calcul des descripteurs")

    with tab2:
        st.header("Analyse de fichiers batch")

        # Instructions
        st.info("""
        📋 **Formats supportés:** Fichiers CSV et Excel (.xlsx)  
        📊 **Limite recommandée:** <1000 molécules pour de bonnes performances  
        ⏱️ **Temps de calcul:** ~1-2 secondes par molécule

        **Colonnes SMILES acceptées:** SMILES, Canonical SMILES, canonical_smiles, smiles, etc.
        """)

        uploaded_file = st.file_uploader("Sélectionner fichier", type=['csv', 'xlsx'])

        if uploaded_file:
            try:
                # Chargement du fichier selon l'extension
                file_extension = uploaded_file.name.split('.')[-1].lower()

                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                elif file_extension in ['xlsx', 'xls']:
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("Format de fichier non supporté")
                    return

                st.success(f"✅ Fichier chargé: {len(df)} lignes détectées")

                # Détection flexible de la colonne SMILES
                possible_smiles_columns = [
                    'SMILES', 'Canonical SMILES', 'canonical_smiles', 'smiles',
                    'Canonical_SMILES', 'CANONICAL_SMILES', 'Smiles',
                    'canonical smiles', 'SMILES_canonical', 'mol_smiles'
                ]

                detected_smiles_col = None
                for col_name in possible_smiles_columns:
                    if col_name in df.columns:
                        detected_smiles_col = col_name
                        break

                if detected_smiles_col:
                    st.success(f"🎯 Colonne SMILES détectée: '{detected_smiles_col}'")

                    # Option pour changer de colonne si l'utilisateur le souhaite
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        smiles_column = st.selectbox(
                            "Colonne SMILES à utiliser:",
                            options=df.columns.tolist(),
                            index=df.columns.tolist().index(detected_smiles_col)
                        )
                    with col2:
                        if smiles_column != detected_smiles_col:
                            st.warning("⚠️ Colonne modifiée")

                else:
                    # Aucune colonne SMILES détectée automatiquement
                    st.warning("⚠️ Aucune colonne SMILES détectée automatiquement")
                    smiles_column = st.selectbox(
                        "Sélectionner manuellement la colonne SMILES:",
                        options=df.columns.tolist(),
                        help="Choisissez la colonne contenant les codes SMILES"
                    )

                # Vérification que la colonne sélectionnée existe et contient des données
                if smiles_column and smiles_column in df.columns:
                    # Vérifier qu'il y a des données non-vides dans cette colonne
                    non_empty_smiles = df[smiles_column].dropna()
                    if len(non_empty_smiles) == 0:
                        st.error(f"❌ La colonne '{smiles_column}' ne contient aucune donnée SMILES valide")
                        return

                    st.write("**Aperçu des données:**")
                    preview_df = df.head(10).copy()
                    # Mettre en évidence la colonne SMILES sélectionnée
                    if len(preview_df.columns) > 10:
                        # Afficher prioritairement la colonne SMILES et quelques autres
                        priority_cols = [smiles_column]
                        other_cols = [col for col in df.columns if col != smiles_column][:9]
                        display_cols = priority_cols + other_cols
                        preview_df = preview_df[display_cols]

                    st.dataframe(preview_df)

                    # Statistiques sur les données SMILES
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total molécules", len(df))
                    with col2:
                        valid_smiles = len(non_empty_smiles)
                        st.metric("SMILES valides", valid_smiles)
                    with col3:
                        empty_smiles = len(df) - valid_smiles
                        st.metric("SMILES vides", empty_smiles)
                    with col4:
                        estimated_time = valid_smiles * 1.5  # secondes
                        st.metric("Temps estimé", f"{estimated_time / 60:.1f} min")

                    if empty_smiles > 0:
                        st.warning(f"⚠️ {empty_smiles} lignes avec SMILES vides seront ignorées")

                    # Options avancées
                    with st.expander("⚙️ Options avancées"):
                        col1, col2 = st.columns(2)
                        with col1:
                            skip_invalid = st.checkbox("Ignorer les SMILES invalides", value=True)
                            save_errors = st.checkbox("Sauvegarder les erreurs", value=True)
                        with col2:
                            batch_size = st.slider("Taille des lots", min_value=10, max_value=100, value=50)
                            show_progress = st.checkbox("Affichage détaillé", value=True)

                    if st.button("🚀 Lancer prédictions batch", type="primary"):
                        model = load_model()
                        if model is None:
                            return

                        results = []
                        errors = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Filtrer les lignes avec SMILES non-vides
                        df_filtered = df.dropna(subset=[smiles_column]).copy()

                        for i, row in df_filtered.iterrows():
                            smiles = row[smiles_column]

                            if show_progress:
                                status_text.text(f'Molécule {i + 1}/{len(df_filtered)}: {str(smiles)[:40]}...')

                            try:
                                # Calculer features et prédire
                                features = calculate_descriptors(smiles)
                                if features:
                                    pred, probs, conf = predict_dili(model, features)
                                    prob_dili = probs[1] if probs is not None and len(probs) > 1 else 0.5
                                    result = 'DILI' if pred == 1 else 'Non-DILI' if pred == 0 else 'Erreur'

                                    # Ajouter toutes les colonnes du fichier original
                                    result_row = row.to_dict()
                                    result_row.update({
                                        'Prediction': result,
                                        'DILI_Probability': prob_dili,
                                        'Confidence': conf,
                                        'Risk_Level': ('Élevé' if prob_dili and prob_dili > 0.7 else
                                                       'Modéré' if prob_dili and prob_dili > 0.3 else
                                                       'Faible' if prob_dili else 'Inconnu')
                                    })
                                    results.append(result_row)

                                else:
                                    # SMILES invalide
                                    if skip_invalid:
                                        error_info = {
                                            'Index': i,
                                            'SMILES': smiles,
                                            'Erreur': 'SMILES invalide',
                                            **row.to_dict()
                                        }
                                        errors.append(error_info)
                                    else:
                                        result_row = row.to_dict()
                                        result_row.update({
                                            'Prediction': 'Erreur SMILES',
                                            'DILI_Probability': None,
                                            'Confidence': None,
                                            'Risk_Level': 'Inconnu'
                                        })
                                        results.append(result_row)

                            except Exception as e:
                                error_info = {
                                    'Index': i,
                                    'SMILES': smiles,
                                    'Erreur': str(e),
                                    **row.to_dict()
                                }
                                errors.append(error_info)

                                if not skip_invalid:
                                    result_row = row.to_dict()
                                    result_row.update({
                                        'Prediction': 'Erreur calcul',
                                        'DILI_Probability': None,
                                        'Confidence': None,
                                        'Risk_Level': 'Inconnu'
                                    })
                                    results.append(result_row)

                            progress_bar.progress((i + 1) / len(df_filtered))

                        # Afficher résultats
                        status_text.text("✅ Analyse terminée!")

                        if results:
                            results_df = pd.DataFrame(results)
                            st.success(f"🎉 Prédictions terminées pour {len(results)} molécules!")

                            # Réorganiser les colonnes pour mettre les résultats en premier
                            result_columns = ['Prediction', 'DILI_Probability', 'Confidence', 'Risk_Level']
                            other_columns = [col for col in results_df.columns if col not in result_columns]
                            ordered_columns = result_columns + other_columns
                            results_df = results_df[ordered_columns]

                            st.dataframe(results_df, use_container_width=True)

                            # Statistiques rapides
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                n_dili = sum(1 for r in results if r['Prediction'] == 'DILI')
                                st.metric("DILI positifs", n_dili)
                            with col2:
                                n_non_dili = sum(1 for r in results if r['Prediction'] == 'Non-DILI')
                                st.metric("Non-DILI", n_non_dili)
                            with col3:
                                if errors:
                                    st.metric("Erreurs", len(errors))
                                else:
                                    st.metric("Erreurs", 0)
                            with col4:
                                valid_conf = [r['Confidence'] for r in results if r['Confidence'] is not None]
                                if valid_conf:
                                    avg_conf = np.mean(valid_conf)
                                    st.metric("Confiance moy.", f"{avg_conf:.1%}")
                                else:
                                    st.metric("Confiance moy.", "N/A")

                            # Téléchargement des résultats
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Télécharger résultats complets",
                                data=csv,
                                file_name=f"dili_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )

                        # Afficher les erreurs si demandé
                        if errors and save_errors:
                            st.subheader("🚨 Erreurs détectées")
                            errors_df = pd.DataFrame(errors)
                            st.dataframe(errors_df, use_container_width=True)

                            # Téléchargement des erreurs
                            csv_errors = errors_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Télécharger rapport d'erreurs",
                                data=csv_errors,
                                file_name=f"dili_errors_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )

                else:
                    st.error("❌ Aucune colonne SMILES sélectionnée ou colonne introuvable")
                    st.write("Colonnes disponibles:", list(df.columns))

            except Exception as e:
                st.error(f"Erreur lecture fichier: {e}")
                st.write("Vérifiez que le fichier est bien formaté et accessible.")

    with tab3:
        st.header("Chimiothèque - Pharmacopée Ouest-Africaine")

        # Information sur SwissADME
        st.info("""
        **ℹ️ Source des propriétés:** Les propriétés physicochimiques et ADMET des molécules ont été calculées 
        avec [SwissADME](http://www.swissadme.ch), un outil web gratuit d'évaluation pharmacocinétique.

        **Référence:** Daina, A., Michielin, O. & Zoete, V. SwissADME: a free web tool to evaluate pharmacokinetics, 
        drug-likeness and medicinal chemistry friendliness of small molecules. *Sci. Rep.* (2017) 7:42717.
        """)

        # Charger la chimiothèque
        pharmacopee_df = load_pharmacopee_database()

        if pharmacopee_df is not None:
            # Statistiques générales
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Molécules totales", len(pharmacopee_df))
            with col2:
                n_plantes = pharmacopee_df['Plante'].nunique()
                st.metric("Plantes distinctes", n_plantes)
            with col3:
                avg_mw = pharmacopee_df['MW'].mean()
                st.metric("MW moyen", f"{avg_mw:.1f} g/mol")
            with col4:
                high_gi = sum(pharmacopee_df['GI absorption'] == 'High')
                st.metric("Absorption GI élevée", f"{high_gi}/{len(pharmacopee_df)}")

            # Onglets secondaires pour la pharmacopée
            sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(
                ["🔍 Exploration", "📊 Visualisations", "⚗️ Prédictions DILI", "💾 Export"])

            with sub_tab1:
                st.subheader("Filtrage et recherche")

                # Filtres
                col1, col2 = st.columns(2)
                with col1:
                    plante_search = st.text_input("Recherche par plante:", placeholder="Ex: ABRUS")
                    molecule_search = st.text_input("Recherche par molécule:", placeholder="Ex: Abruquinone")

                with col2:
                    mw_range = st.slider("Poids moléculaire (g/mol)",
                                         float(pharmacopee_df['MW'].min()),
                                         float(pharmacopee_df['MW'].max()),
                                         (float(pharmacopee_df['MW'].min()), float(pharmacopee_df['MW'].max())))

                    logp_options = pharmacopee_df['Consensus Log P'].dropna()
                    logp_range = st.slider("Log P consensuel",
                                           float(logp_options.min()),
                                           float(logp_options.max()),
                                           (float(logp_options.min()), float(logp_options.max())))

                col3, col4 = st.columns(2)
                with col3:
                    gi_options = ['Tous'] + list(pharmacopee_df['GI absorption'].unique())
                    gi_absorption = st.selectbox("Absorption GI:", gi_options)

                with col4:
                    lipinski_max = st.slider("Max violations Lipinski:", 0, 5, 5)

                # Appliquer les filtres
                filters = {
                    'mw_range': mw_range,
                    'logp_range': logp_range,
                    'gi_absorption': gi_absorption,
                    'lipinski_violations': lipinski_max,
                    'plante_search': plante_search,
                    'molecule_search': molecule_search
                }

                filtered_df = filter_pharmacopee_data(pharmacopee_df, filters)

                st.write(f"**{len(filtered_df)} molécules trouvées**")

                # Affichage des résultats
                if len(filtered_df) > 0:
                    # Sélectionner les colonnes principales pour l'affichage
                    display_cols = ['Plante', 'Molecule', 'Formula', 'MW', 'Consensus Log P',
                                    'TPSA', 'GI absorption', 'Lipinski #violations', 'Bioavailability Score']

                    display_df = filtered_df[display_cols].copy()
                    display_df = display_df.round(2)

                    # Utiliser st.data_editor pour permettre la sélection
                    edited_df = st.data_editor(
                        display_df,
                        hide_index=True,
                        use_container_width=True,
                        num_rows="dynamic"
                    )

                    # Détails d'une molécule sélectionnée
                    selected_idx = st.selectbox("Sélectionner une molécule pour détails:",
                                                range(len(filtered_df)),
                                                format_func=lambda
                                                    x: f"{filtered_df.iloc[x]['Molecule']} ({filtered_df.iloc[x]['Plante']})")

                    if selected_idx is not None:
                        mol_info = filtered_df.iloc[selected_idx]

                        col1, col2 = st.columns([1, 1])

                        with col1:
                            st.subheader(f"Détails - {mol_info['Molecule']}")
                            st.write(f"**Plante:** {mol_info['Plante']}")
                            st.write(f"**Formule:** {mol_info['Formula']}")
                            st.write(f"**SMILES:** {mol_info['Canonical SMILES']}")
                            st.write(f"**MW:** {mol_info['MW']:.2f} g/mol")
                            st.write(f"**Log P:** {mol_info['Consensus Log P']:.2f}")
                            st.write(f"**TPSA:** {mol_info['TPSA']:.2f} Å²")

                        with col2:
                            # Afficher la structure moléculaire si possible
                            mol_img = draw_molecule(mol_info['Canonical SMILES'], size=(300, 200))
                            if mol_img:
                                st.image(mol_img, caption="Structure 2D")

                            # Propriétés ADMET
                            st.subheader("Propriétés ADMET")
                            st.write(f"**Absorption GI:** {mol_info['GI absorption']}")
                            st.write(f"**Perméabilité BBB:** {mol_info['BBB permeant']}")
                            st.write(f"**Substrat Pgp:** {mol_info['Pgp substrate']}")
                            st.write(f"**Score biodisponibilité:** {mol_info['Bioavailability Score']:.2f}")

            with sub_tab2:
                st.subheader("Visualisations des propriétés")

                # Graphiques de la pharmacopée
                fig = create_pharmacopee_visualizations(pharmacopee_df)
                st.pyplot(fig)

                # Graphiques interactifs supplémentaires
                st.subheader("Analyse interactive")

                col1, col2 = st.columns(2)
                with col1:
                    x_axis = st.selectbox("Axe X:", ['MW', 'Consensus Log P', 'TPSA', 'Bioavailability Score'])
                with col2:
                    y_axis = st.selectbox("Axe Y:", ['TPSA', 'MW', 'Consensus Log P', 'Bioavailability Score'])

                if x_axis != y_axis:
                    fig_scatter, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(pharmacopee_df[x_axis].dropna(),
                                         pharmacopee_df[y_axis].dropna(),
                                         c=pharmacopee_df['Lipinski #violations'].dropna(),
                                         cmap='RdYlGn_r', alpha=0.6, s=50)
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel(y_axis)
                    ax.set_title(f'{x_axis} vs {y_axis} (coloré par violations Lipinski)')
                    plt.colorbar(scatter, label='Violations Lipinski')
                    st.pyplot(fig_scatter)

            with sub_tab3:
                st.subheader("Prédictions DILI sur la pharmacopée")

                st.info("Sélectionnez des molécules de la pharmacopée pour prédire leur toxicité hépatique")

                # Options de sélection
                selection_mode = st.radio("Mode de sélection:", ["Molécules individuelles", "Toute la base"])

                if selection_mode == "Molécules individuelles":
                    # Sélection multiple
                    selected_molecules = st.multiselect(
                        "Sélectionner molécules:",
                        options=range(len(pharmacopee_df)),
                        format_func=lambda
                            x: f"{pharmacopee_df.iloc[x]['Molecule']} ({pharmacopee_df.iloc[x]['Plante']})",
                        max_selections=20  # Limiter pour éviter surcharge
                    )

                    if selected_molecules and st.button("Prédire DILI pour sélection", type="primary"):
                        model = load_model()
                        if model:
                            results = []
                            progress_bar = st.progress(0)

                            for i, idx in enumerate(selected_molecules):
                                mol_info = pharmacopee_df.iloc[idx]
                                smiles = mol_info['Canonical SMILES']

                                # Calculer features et prédire
                                features = calculate_descriptors(smiles)
                                if features:
                                    pred, probs, conf = predict_dili(model, features)
                                    prob_dili = probs[1] if probs is not None and len(probs) > 1 else 0.5
                                else:
                                    pred, prob_dili, conf = None, None, None

                                results.append({
                                    'Plante': mol_info['Plante'],
                                    'Molecule': mol_info['Molecule'],
                                    'SMILES': smiles,
                                    'MW': mol_info['MW'],
                                    'Prediction_DILI': 'DILI' if pred == 1 else 'Non-DILI' if pred == 0 else 'Erreur',
                                    'Probabilite_DILI': prob_dili,
                                    'Confiance': conf,
                                    'GI_absorption': mol_info['GI absorption'],
                                    'Bioavailability_Score': mol_info['Bioavailability Score']
                                })

                                progress_bar.progress((i + 1) / len(selected_molecules))

                            results_df = pd.DataFrame(results)
                            st.dataframe(results_df, use_container_width=True)

                            # Statistiques rapides
                            n_dili = sum(1 for r in results if r['Prediction_DILI'] == 'DILI')
                            st.metric("Molécules prédites DILI+", f"{n_dili}/{len(results)}")

                            # Téléchargement
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "Télécharger résultats DILI",
                                csv,
                                f"pharmacopee_dili_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv"
                            )

                elif selection_mode == "Toute la base":
                    st.warning(
                        f"Cela va traiter {len(pharmacopee_df)} molécules (~{len(pharmacopee_df) * 1.5 / 60:.1f} minutes)")

                    if st.button("Prédire DILI pour toute la pharmacopée", type="secondary"):
                        model = load_model()
                        if model:
                            st.write("Traitement en cours... Cela peut prendre plusieurs minutes.")

                            # Traitement par batch pour affichage progressif
                            batch_size = 10
                            all_results = []

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            for batch_start in range(0, len(pharmacopee_df), batch_size):
                                batch_end = min(batch_start + batch_size, len(pharmacopee_df))
                                batch_df = pharmacopee_df.iloc[batch_start:batch_end]

                                for i, (_, mol_info) in enumerate(batch_df.iterrows()):
                                    current_idx = batch_start + i
                                    status_text.text(
                                        f'Molécule {current_idx + 1}/{len(pharmacopee_df)}: {mol_info["Molecule"]}')

                                    smiles = mol_info['Canonical SMILES']
                                    features = calculate_descriptors(smiles)

                                    if features:
                                        pred, probs, conf = predict_dili(model, features)
                                        prob_dili = probs[1] if probs is not None and len(probs) > 1 else 0.5
                                    else:
                                        pred, prob_dili, conf = None, None, None

                                    all_results.append({
                                        'Plante': mol_info['Plante'],
                                        'Molecule': mol_info['Molecule'],
                                        'MW': mol_info['MW'],
                                        'Prediction_DILI': 'DILI' if pred == 1 else 'Non-DILI' if pred == 0 else 'Erreur',
                                        'Probabilite_DILI': prob_dili,
                                        'Confiance': conf,
                                        'GI_absorption': mol_info['GI absorption']
                                    })

                                    progress_bar.progress((current_idx + 1) / len(pharmacopee_df))

                            # Afficher résultats complets
                            results_df = pd.DataFrame(all_results)
                            st.success(f"Analyse terminée pour {len(results_df)} molécules!")

                            # Statistiques
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                n_dili = sum(results_df['Prediction_DILI'] == 'DILI')
                                st.metric("Prédictions DILI+", f"{n_dili}")
                            with col2:
                                n_non_dili = sum(results_df['Prediction_DILI'] == 'Non-DILI')
                                st.metric("Prédictions DILI-", f"{n_non_dili}")
                            with col3:
                                avg_conf = results_df['Confiance'].dropna().mean()
                                st.metric("Confiance moyenne", f"{avg_conf:.1%}")

                            st.dataframe(results_df, use_container_width=True)

                            # Téléchargement complet
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "Télécharger toutes les prédictions DILI",
                                csv,
                                f"pharmacopee_complete_dili_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv"
                            )

            with sub_tab4:
                st.subheader("Export et téléchargement")

                st.write("**Base de données complète:**")
                st.write(f"Contient {len(pharmacopee_df)} molécules avec {len(pharmacopee_df.columns)} propriétés")

                # Options d'export
                export_format = st.selectbox("Format d'export:", ["CSV", "Excel", "JSON"])

                if export_format == "CSV":
                    csv_data = pharmacopee_df.to_csv(index=False)
                    st.download_button(
                        "Télécharger base complète (CSV)",
                        csv_data,
                        f"pharmacopee_ouest_africaine_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                        "text/csv"
                    )

                elif export_format == "Excel":
                    # Créer fichier Excel en mémoire
                    from io import BytesIO
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        pharmacopee_df.to_excel(writer, sheet_name='Molecules', index=False)

                        # Statistiques sur un autre onglet
                        stats_data = {
                            'Propriété': ['Nombre total molécules', 'Nombre plantes', 'MW moyen', 'MW médian',
                                          'LogP moyen', 'TPSA moyen', 'GI absorption haute',
                                          'Violations Lipinski moyennes'],
                            'Valeur': [len(pharmacopee_df), pharmacopee_df['Plante'].nunique(),
                                       pharmacopee_df['MW'].mean(), pharmacopee_df['MW'].median(),
                                       pharmacopee_df['Consensus Log P'].mean(), pharmacopee_df['TPSA'].mean(),
                                       sum(pharmacopee_df['GI absorption'] == 'High'),
                                       pharmacopee_df['Lipinski #violations'].mean()]
                        }
                        pd.DataFrame(stats_data).to_excel(writer, sheet_name='Statistiques', index=False)

                    st.download_button(
                        "Télécharger base complète (Excel)",
                        buffer.getvalue(),
                        f"pharmacopee_ouest_africaine_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                # Sélection de colonnes pour export personnalisé
                st.subheader("Export personnalisé")
                selected_cols = st.multiselect(
                    "Sélectionner colonnes à exporter:",
                    options=pharmacopee_df.columns.tolist(),
                    default=['Plante', 'Molecule', 'Canonical SMILES', 'MW', 'Consensus Log P', 'TPSA']
                )

                if selected_cols:
                    custom_df = pharmacopee_df[selected_cols]
                    csv_custom = custom_df.to_csv(index=False)
                    st.download_button(
                        "Télécharger sélection personnalisée",
                        csv_custom,
                        f"pharmacopee_custom_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                        "text/csv"
                    )
        else:
            st.error("Impossible de charger la base de données de pharmacopée.")

    with tab4:
        st.header("À propos du modèle DILI")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            ### 🧬 Modèle de prédiction DILI (Drug-Induced Liver Injury)

            Ce modèle de machine learning prédit la toxicité hépatique potentielle des molécules chimiques 
            basé sur leur structure moléculaire, développé dans le contexte de recherche sur la pharmacopée ouest-africaine.

            **🔬 Caractéristiques techniques:**
            - **Algorithme:** Stacking Classifier (ensemble de modèles)
            - **Modèles de base:** Random Forest, SVM (linéaire/RBF), k-NN
            - **Meta-estimateur:** Régression logistique
            - **Features:** 36 descripteurs physicochimiques + 1024 Morgan fingerprints (R=2)
            - **Dataset:** 945 molécules d'entraînement
            - **Validation:** Nested cross-validation + validation externe rigoureuse

            **📊 Performance:**
            - F1-score externe: 75.6%
            - AUC-ROC externe: 79.5%
            - Spécificité: 84.6%
            - Sensibilité: 76.9%

            **🌿 Chimiothèque pharmacopée:**
            - 191 molécules de plantes médicinales ouest-africaines
            - 50 propriétés physicochimiques et ADMET par molécule
            - Propriétés calculées avec SwissADME
            - Données issues de sources ethnopharmacologiques validées

            **⚗️ Descripteurs utilisés:**
            Les 36 descripteurs incluent des propriétés comme la lipophilicité (XLogP), 
            les compteurs de liaisons hydrogène, les descripteurs de surface moléculaire, 
            et divers fragments fonctionnels.
            """)

        with col2:
            st.markdown("""
            ### 📚 Références

            **Méthodologie:**
            - Nested Cross-Validation
            - Stratégies comparatives
            - Validation externe rigoureuse
            - Consensus hyperparamètres

            **⚠️ Limitations:**
            - Outil de recherche uniquement
            - Dataset externe limité (156 mol.)
            - Domaine d'applicabilité à considérer
            - Performance modeste (75.6% F1)

            **🌍 Contexte régional:**
            Développé pour la recherche sur les 
            substances bioactives de la 
            pharmacopée traditionnelle 
            ouest-africaine.

            ### 🏥 Avertissement médical

            Ce modèle est destiné exclusivement 
            à la recherche académique et ne doit 
            **jamais** être utilisé pour des 
            décisions cliniques ou thérapeutiques.
            """)

        # Informations techniques détaillées
        with st.expander("🔧 Détails techniques"):
            st.markdown("""
            **Pipeline de preprocessing:**
            - StandardScaler sur les descripteurs physicochimiques
            - Passthrough sur les Morgan fingerprints (données binaires)
            - Gestion des valeurs manquantes (fillna=0)

            **Modèles ensemblés dans le Stacking:**
            - Random Forest (class_weight='balanced')
            - SVM Linéaire et RBF (class_weight='balanced')
            - k-Nearest Neighbors
            - Meta-estimateur: Logistic Regression

            **Hyperparamètres optimisés par GridSearchCV avec consensus robuste**

            **Base de données pharmacopée:**
            - Source: Collecte ethnopharmacologique systématique
            - Validation: Structures chimiques vérifiées
            - Propriétés: Calculées via outils chémoinformatiques
            - Couverture: Plantes médicinales CEDEAO
            """)

        # Contact/Support
        st.markdown("""
        ---
        **🔧 Support technique:** Pour questions ou problèmes techniques  
        **📖 Documentation:** Voir rapport détaillé pour méthodologie complète  
        **🌿 Base pharmacopée:** 191 molécules de plantes ouest-africaines  
        **📄 Version:** 1.0 (Août 2025)  
        **🎯 Contexte:** Recherche pharmacopée ouest-africaine - OOAS
        """)


if __name__ == "__main__":
    main()