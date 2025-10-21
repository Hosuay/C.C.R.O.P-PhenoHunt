"""
Scientific Visualization Tools
Enhanced visualizations with uncertainty quantification
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ScientificVisualizer:
    """
    Create publication-quality visualizations with uncertainty.
    """

    @staticmethod
    def plot_chemical_profile_with_uncertainty(
        profile: pd.Series,
        uncertainty: Optional[pd.Series] = None,
        title: str = "Chemical Profile",
        config: Optional[Dict] = None
    ) -> go.Figure:
        """
        Plot chemical profile with error bars.

        Args:
            profile: Mean chemical values
            uncertainty: Standard deviations
            title: Plot title
            config: Configuration dict

        Returns:
            Plotly figure
        """
        # Separate cannabinoids and terpenes
        cannabinoid_cols = [col for col in profile.index if any(
            cann in col for cann in ['thc', 'cbd', 'cbg', 'cbc', 'cbda', 'thcv', 'cbn', 'delta8', 'thca']
        )]
        terpene_cols = [col for col in profile.index if any(
            terp in col for terp in ['myrcene', 'limonene', 'pinene', 'linalool',
                                     'caryophyllene', 'humulene', 'terpinolene',
                                     'ocimene', 'camphene', 'bisabolol']
        )]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Cannabinoids", "Terpenes"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        # Plot cannabinoids
        for col in cannabinoid_cols:
            compound_name = col.replace('_pct', '').upper()
            value = profile[col]
            error = uncertainty[col] if uncertainty is not None else 0

            fig.add_trace(
                go.Bar(
                    name=compound_name,
                    x=[compound_name],
                    y=[value],
                    error_y=dict(type='data', array=[error], visible=True),
                    text=[f"{value:.2f}±{error:.2f}%"],
                    textposition='auto'
                ),
                row=1, col=1
            )

        # Plot terpenes
        for col in terpene_cols:
            compound_name = col.replace('_pct', '').upper()
            value = profile[col]
            error = uncertainty[col] if uncertainty is not None else 0

            fig.add_trace(
                go.Bar(
                    name=compound_name,
                    x=[compound_name],
                    y=[value],
                    error_y=dict(type='data', array=[error], visible=True),
                    text=[f"{value:.2f}±{error:.2f}%"],
                    textposition='auto'
                ),
                row=1, col=2
            )

        fig.update_layout(
            title_text=title,
            title_x=0.5,
            showlegend=False,
            height=500,
            font=dict(size=12)
        )

        fig.update_yaxes(title_text="Concentration (%)", row=1, col=1)
        fig.update_yaxes(title_text="Concentration (%)", row=1, col=2)

        return fig

    @staticmethod
    def plot_effect_predictions(
        effect_results: pd.DataFrame,
        strain_name: str = "Candidate"
    ) -> go.Figure:
        """
        Plot therapeutic effect predictions with confidence intervals.

        Args:
            effect_results: DataFrame with effect predictions
            strain_name: Name of strain

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Sort by probability
        effect_results = effect_results.sort_values('probability', ascending=True)

        # Color code by confidence
        colors = [
            'rgb(34, 139, 34)' if row['confidence_met'] else 'rgb(255, 140, 0)'
            for _, row in effect_results.iterrows()
        ]

        # Create horizontal bar chart
        fig.add_trace(go.Bar(
            y=effect_results['effect'],
            x=effect_results['probability'],
            orientation='h',
            marker=dict(color=colors),
            error_x=dict(
                type='data',
                symmetric=False,
                array=effect_results['upper_bound'] - effect_results['probability'],
                arrayminus=effect_results['probability'] - effect_results['lower_bound']
            ),
            text=[f"{p:.1%} ± {u:.1%}"
                  for p, u in zip(effect_results['probability'], effect_results['uncertainty'])],
            textposition='auto'
        ))

        # Add threshold line
        if len(effect_results) > 0:
            threshold = effect_results.iloc[0]['probability']  # Just for reference
            fig.add_vline(
                x=0.6,
                line_dash="dash",
                line_color="gray",
                annotation_text="Typical Threshold"
            )

        fig.update_layout(
            title=f"Therapeutic Effect Predictions: {strain_name}",
            title_x=0.5,
            xaxis_title="Probability",
            yaxis_title="Therapeutic Effect",
            height=400,
            font=dict(size=12),
            showlegend=False
        )

        fig.update_xaxes(range=[0, 1])

        return fig

    @staticmethod
    def plot_breeding_comparison(
        parent1_profile: pd.Series,
        parent2_profile: pd.Series,
        offspring_profile: pd.Series,
        parent1_name: str = "Parent 1",
        parent2_name: str = "Parent 2",
        offspring_name: str = "Offspring"
    ) -> go.Figure:
        """
        Radar chart comparing parents and offspring.

        Args:
            parent1_profile: First parent profile
            parent2_profile: Second parent profile
            offspring_profile: Offspring profile
            parent1_name, parent2_name, offspring_name: Strain names

        Returns:
            Plotly figure
        """
        # Select key compounds for visualization
        key_compounds = [
            'thc_pct', 'cbd_pct', 'cbg_pct',
            'myrcene_pct', 'limonene_pct', 'caryophyllene_pct'
        ]

        available_compounds = [c for c in key_compounds if c in parent1_profile.index]

        if len(available_compounds) == 0:
            logger.warning("No compounds available for radar plot")
            return go.Figure()

        categories = [c.replace('_pct', '').upper() for c in available_compounds]

        fig = go.Figure()

        # Parent 1
        fig.add_trace(go.Scatterpolar(
            r=[parent1_profile[c] for c in available_compounds],
            theta=categories,
            fill='toself',
            name=parent1_name,
            line=dict(color='#667eea', width=2)
        ))

        # Parent 2
        fig.add_trace(go.Scatterpolar(
            r=[parent2_profile[c] for c in available_compounds],
            theta=categories,
            fill='toself',
            name=parent2_name,
            line=dict(color='#764ba2', width=2)
        ))

        # Offspring
        fig.add_trace(go.Scatterpolar(
            r=[offspring_profile[c] for c in available_compounds],
            theta=categories,
            fill='toself',
            name=offspring_name,
            line=dict(color='#43e97b', width=3)
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(parent1_profile[available_compounds].max(),
                                  parent2_profile[available_compounds].max(),
                                  offspring_profile[available_compounds].max()) * 1.1]
                )
            ),
            showlegend=True,
            title="Parent-Offspring Comparison",
            title_x=0.5,
            height=500
        )

        return fig

    @staticmethod
    def plot_latent_space_interpolation(
        interpolated_profiles: np.ndarray,
        parent1_name: str,
        parent2_name: str,
        feature_names: List[str]
    ) -> go.Figure:
        """
        Visualize chemical profile interpolation in latent space.

        Shows how chemical composition transitions from parent1 to parent2.

        Args:
            interpolated_profiles: Array of interpolated profiles [n_steps, n_features]
            parent1_name: First parent name
            parent2_name: Second parent name
            feature_names: List of feature names

        Returns:
            Plotly figure
        """
        n_steps = interpolated_profiles.shape[0]

        # Select a few key compounds to visualize
        key_indices = [0, 1, 2, 4, 5]  # THC, CBD, CBG, Myrcene, Limonene (typically)
        key_features = [feature_names[i] for i in key_indices if i < len(feature_names)]

        fig = go.Figure()

        for i, feature in enumerate(key_features):
            if i < interpolated_profiles.shape[1]:
                values = interpolated_profiles[:, i]

                fig.add_trace(go.Scatter(
                    x=list(range(n_steps)),
                    y=values,
                    mode='lines+markers',
                    name=feature.replace('_pct', '').upper(),
                    line=dict(width=2),
                    marker=dict(size=6)
                ))

        fig.update_layout(
            title=f"Chemical Profile Interpolation: {parent1_name} → {parent2_name}",
            title_x=0.5,
            xaxis_title="Interpolation Step",
            yaxis_title="Concentration (%)",
            height=500,
            hovermode='x unified',
            font=dict(size=12)
        )

        # Add annotations for parents
        fig.add_annotation(
            x=0, y=0,
            text=parent1_name,
            showarrow=False,
            yshift=-40
        )

        fig.add_annotation(
            x=n_steps-1, y=0,
            text=parent2_name,
            showarrow=False,
            yshift=-40
        )

        return fig

    @staticmethod
    def plot_uncertainty_heatmap(
        profiles_df: pd.DataFrame,
        uncertainties_df: pd.DataFrame,
        title: str = "Uncertainty Heatmap"
    ) -> go.Figure:
        """
        Create heatmap showing prediction uncertainty across strains and compounds.

        Args:
            profiles_df: DataFrame with chemical profiles
            uncertainties_df: DataFrame with uncertainties
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Mean Values", "Uncertainty (Std Dev)"),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}]]
        )

        # Mean values heatmap
        fig.add_trace(
            go.Heatmap(
                z=profiles_df.values,
                x=profiles_df.columns,
                y=profiles_df.index,
                colorscale='Viridis',
                colorbar=dict(x=0.45, title="Mean %")
            ),
            row=1, col=1
        )

        # Uncertainty heatmap
        fig.add_trace(
            go.Heatmap(
                z=uncertainties_df.values,
                x=uncertainties_df.columns,
                y=uncertainties_df.index,
                colorscale='Reds',
                colorbar=dict(x=1.02, title="Std Dev %")
            ),
            row=1, col=2
        )

        fig.update_layout(
            title_text=title,
            title_x=0.5,
            height=400,
            font=dict(size=10)
        )

        return fig

    @staticmethod
    def plot_validation_metrics(
        actual_values: np.ndarray,
        predicted_values: np.ndarray,
        compound_names: List[str]
    ) -> go.Figure:
        """
        Plot validation metrics comparing actual vs predicted values.

        Args:
            actual_values: Actual compound values
            predicted_values: Predicted compound values
            compound_names: List of compound names

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Actual vs Predicted", "Residual Plot"),
            specs=[[{"type": "scatter"}, {"type": "scatter"}]]
        )

        # Scatter plot: actual vs predicted
        fig.add_trace(
            go.Scatter(
                x=actual_values.flatten(),
                y=predicted_values.flatten(),
                mode='markers',
                marker=dict(size=8, opacity=0.6, color='#667eea'),
                text=compound_names * len(actual_values),
                hovertemplate='<b>%{text}</b><br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}',
                name='Data'
            ),
            row=1, col=1
        )

        # Add diagonal line (perfect prediction)
        max_val = max(actual_values.max(), predicted_values.max())
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Perfect Prediction'
            ),
            row=1, col=1
        )

        # Residual plot
        residuals = actual_values.flatten() - predicted_values.flatten()
        fig.add_trace(
            go.Scatter(
                x=predicted_values.flatten(),
                y=residuals,
                mode='markers',
                marker=dict(size=8, opacity=0.6, color='#43e97b'),
                text=compound_names * len(actual_values),
                hovertemplate='<b>%{text}</b><br>Predicted: %{x:.2f}<br>Residual: %{y:.2f}',
                name='Residuals'
            ),
            row=1, col=2
        )

        # Add zero line for residuals
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

        fig.update_xaxes(title_text="Actual Value (%)", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Value (%)", row=1, col=1)
        fig.update_xaxes(title_text="Predicted Value (%)", row=1, col=2)
        fig.update_yaxes(title_text="Residual (%)", row=1, col=2)

        fig.update_layout(
            title_text="Model Validation",
            title_x=0.5,
            height=400,
            showlegend=True,
            font=dict(size=12)
        )

        return fig
