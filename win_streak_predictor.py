#!/usr/bin/env python3
"""
Win Streak Transition Predictor
Predicts how many more games needed before transitioning from loss streak to win streak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from performance_window_analyzer import PerformanceWindowAnalyzer, PerformanceWindow
from scipy import stats
import seaborn as sns

@dataclass
class LossStreakPattern:
    """Pattern data for a historical loss streak"""
    duration: int  # Total matches in loss streak
    avg_win_rate: float  # Average win rate during streak
    start_date: datetime
    end_date: datetime
    transition_type: str  # How it ended: 'to_win_streak', 'to_neutral'

class WinStreakPredictor:
    """Predicts when current loss streak will end and win streak will begin"""
    
    def __init__(self, player_id: str):
        self.player_id = player_id
        self.analyzer = PerformanceWindowAnalyzer(player_id)
        self.historical_loss_streaks: List[LossStreakPattern] = []
        self.current_loss_streak_duration = 0
        self.current_win_rate = 0.0
        
    def analyze_historical_patterns(self) -> Dict:
        """Analyze historical loss streak patterns to build prediction model"""
        # Fetch data and identify performance windows
        self.analyzer.fetch_matches(days_back=2000)
        windows = self.analyzer.identify_performance_windows()
        
        if not windows:
            return {'error': 'No performance windows found'}
            
        # Extract loss streak patterns
        loss_streaks = [w for w in windows if w.window_type == 'loss_streak']
        
        for i, loss_streak in enumerate(loss_streaks):
            # Determine how this loss streak ended
            if i < len(windows) - 1:
                next_window = windows[i + 1]
                transition_type = f"to_{next_window.window_type}"
            else:
                transition_type = "ongoing"  # Current streak
                
            pattern = LossStreakPattern(
                duration=loss_streak.match_count,
                avg_win_rate=loss_streak.win_rate,
                start_date=loss_streak.start_date,
                end_date=loss_streak.end_date,
                transition_type=transition_type
            )
            
            self.historical_loss_streaks.append(pattern)
            
        # Get current status
        current_status = self.analyzer.get_current_performance_status()
        if current_status.get('current_status') == 'LOSS_STREAK':
            self.current_loss_streak_duration = current_status.get('matches_in_current_window', 0)
            self.current_win_rate = current_status.get('recent_win_rate', 0.0)
        
        return {
            'total_loss_streaks': len(self.historical_loss_streaks),
            'completed_loss_streaks': len([ls for ls in self.historical_loss_streaks if ls.transition_type != 'ongoing']),
            'current_duration': self.current_loss_streak_duration,
            'current_win_rate': self.current_win_rate
        }
    
    def predict_remaining_games(self) -> Dict:
        """Predict how many more games until win streak begins"""
        if not self.historical_loss_streaks:
            return {'error': 'No historical data available'}
            
        # Filter completed loss streaks (exclude ongoing current one)
        completed_streaks = [ls for ls in self.historical_loss_streaks if ls.transition_type != 'ongoing']
        
        if len(completed_streaks) < 2:
            return {'error': 'Need at least 2 completed loss streaks for prediction'}
            
        # Analyze historical durations
        durations = [ls.duration for ls in completed_streaks]
        win_rates = [ls.avg_win_rate for ls in completed_streaks]
        
        # Statistical analysis of loss streak durations
        mean_duration = np.mean(durations)
        median_duration = np.median(durations)
        std_duration = np.std(durations)
        min_duration = np.min(durations)
        max_duration = np.max(durations)
        
        # Current streak analysis
        current_duration = self.current_loss_streak_duration
        current_win_rate = self.current_win_rate
        
        # Prediction methods
        predictions = {}
        
        # Method 1: Statistical average
        remaining_by_mean = max(0, mean_duration - current_duration)
        predictions['statistical_mean'] = {
            'remaining_games': int(remaining_by_mean),
            'confidence': 'medium',
            'method': f'Based on historical average of {mean_duration:.1f} games'
        }
        
        # Method 2: Median-based (more robust to outliers)
        remaining_by_median = max(0, median_duration - current_duration)
        predictions['statistical_median'] = {
            'remaining_games': int(remaining_by_median),
            'confidence': 'high',
            'method': f'Based on historical median of {median_duration:.1f} games'
        }
        
        # Method 3: Percentile-based prediction
        # Where does current duration fall in historical distribution?
        percentile = stats.percentileofscore(durations, current_duration)
        
        if percentile >= 75:
            # Current streak is longer than 75% of historical streaks
            remaining_percentile = max(0, np.percentile(durations, 90) - current_duration)
            confidence = 'low'
            note = 'Current streak is unusually long'
        elif percentile >= 50:
            remaining_percentile = max(0, np.percentile(durations, 75) - current_duration)
            confidence = 'medium'
            note = 'Current streak is above average'
        else:
            remaining_percentile = max(0, np.percentile(durations, 60) - current_duration)
            confidence = 'high'
            note = 'Current streak is typical length'
            
        predictions['percentile_based'] = {
            'remaining_games': int(remaining_percentile),
            'confidence': confidence,
            'method': f'Based on percentile analysis ({percentile:.1f}th percentile)',
            'note': note
        }
        
        # Method 4: Win rate similarity matching
        # Find historical streaks with similar win rates
        similar_streaks = []
        win_rate_tolerance = 0.05  # 5% tolerance
        
        for streak in completed_streaks:
            if abs(streak.avg_win_rate - current_win_rate) <= win_rate_tolerance:
                similar_streaks.append(streak)
                
        if similar_streaks:
            similar_durations = [s.duration for s in similar_streaks]
            similar_mean = np.mean(similar_durations)
            remaining_similar = max(0, similar_mean - current_duration)
            
            predictions['win_rate_matching'] = {
                'remaining_games': int(remaining_similar),
                'confidence': 'high' if len(similar_streaks) >= 3 else 'medium',
                'method': f'Based on {len(similar_streaks)} streaks with similar win rate ({current_win_rate:.1%})',
                'similar_streaks_found': len(similar_streaks)
            }
        
        # Method 5: Trend analysis (if current streak is getting better/worse)
        if current_duration >= 10:
            # Analyze recent trend within current streak
            recent_matches = self.analyzer.matches[-10:]  # Last 10 matches
            recent_wins = sum(1 for m in recent_matches if m.win)
            recent_trend_wr = recent_wins / len(recent_matches)
            
            if recent_trend_wr > current_win_rate + 0.05:  # Improving
                trend_adjustment = -3  # End sooner
                trend_note = "Recent performance improving"
            elif recent_trend_wr < current_win_rate - 0.05:  # Getting worse
                trend_adjustment = +5  # Take longer
                trend_note = "Recent performance declining"
            else:
                trend_adjustment = 0
                trend_note = "Recent performance stable"
                
            trend_prediction = max(0, remaining_by_median + trend_adjustment)
            
            predictions['trend_adjusted'] = {
                'remaining_games': int(trend_prediction),
                'confidence': 'medium',
                'method': 'Median prediction adjusted for recent trend',
                'trend_note': trend_note,
                'recent_win_rate': recent_trend_wr
            }
        
        # Ensemble prediction (weighted average)
        valid_predictions = [p for p in predictions.values() if p['remaining_games'] >= 0]
        
        if valid_predictions:
            # Weight by confidence
            weights = {'high': 3, 'medium': 2, 'low': 1}
            weighted_sum = sum(p['remaining_games'] * weights[p['confidence']] for p in valid_predictions)
            total_weight = sum(weights[p['confidence']] for p in valid_predictions)
            
            ensemble_prediction = int(weighted_sum / total_weight)
            
            predictions['ensemble'] = {
                'remaining_games': ensemble_prediction,
                'confidence': 'high',
                'method': 'Weighted average of all prediction methods',
                'component_predictions': len(valid_predictions)
            }
        
        # Risk assessment
        risk_factors = []
        if current_duration > max_duration:
            risk_factors.append("Current streak is longer than any historical streak")
        if current_win_rate < min(win_rates):
            risk_factors.append("Current win rate is lower than historical loss streaks")
        if percentile > 90:
            risk_factors.append("Current streak duration is in top 10% historically")
            
        return {
            'current_status': {
                'duration': current_duration,
                'win_rate': current_win_rate,
                'percentile': percentile
            },
            'historical_context': {
                'completed_streaks': len(completed_streaks),
                'mean_duration': mean_duration,
                'median_duration': median_duration,
                'std_duration': std_duration,
                'min_duration': min_duration,
                'max_duration': max_duration,
                'duration_range': f"{min_duration}-{max_duration} games"
            },
            'predictions': predictions,
            'risk_factors': risk_factors,
            'recommendation': self._generate_recommendation(predictions, risk_factors)
        }
    
    def _generate_recommendation(self, predictions: Dict, risk_factors: List[str]) -> str:
        """Generate recommendation based on predictions"""
        if 'ensemble' in predictions:
            remaining = predictions['ensemble']['remaining_games']
        elif 'statistical_median' in predictions:
            remaining = predictions['statistical_median']['remaining_games']
        else:
            return "Insufficient data for reliable prediction"
            
        if remaining <= 0:
            return "🎯 BREAKTHROUGH IMMINENT: You should transition to win streak very soon!"
        elif remaining <= 5:
            return f"🟡 ALMOST THERE: Approximately {remaining} more games until win streak begins"
        elif remaining <= 15:
            return f"🔵 PATIENCE NEEDED: Approximately {remaining} more games until win streak begins"
        else:
            return f"🔴 LONG ROAD AHEAD: Approximately {remaining} more games until win streak begins"
    
    def create_prediction_visualization(self, prediction_results: Dict, output_dir: str = ".") -> None:
        """Create visualizations for the prediction analysis"""
        if 'error' in prediction_results:
            print("Cannot create visualization: " + prediction_results['error'])
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Win Streak Transition Prediction Analysis', fontsize=16)
        
        # 1. Historical loss streak durations
        completed_streaks = [ls for ls in self.historical_loss_streaks if ls.transition_type != 'ongoing']
        durations = [ls.duration for ls in completed_streaks]
        
        if durations:
            axes[0, 0].hist(durations, bins=max(3, len(durations)//2), alpha=0.7, color='red', edgecolor='black')
            axes[0, 0].axvline(self.current_loss_streak_duration, color='blue', linestyle='--', linewidth=2, label=f'Current: {self.current_loss_streak_duration}')
            axes[0, 0].axvline(np.mean(durations), color='green', linestyle='-', linewidth=2, label=f'Historical Avg: {np.mean(durations):.1f}')
            axes[0, 0].set_title('Historical Loss Streak Durations')
            axes[0, 0].set_xlabel('Duration (matches)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
        
        # 2. Prediction methods comparison
        if 'predictions' in prediction_results:
            methods = []
            remaining_games = []
            confidences = []
            
            for method, pred in prediction_results['predictions'].items():
                if method != 'ensemble':  # Show ensemble separately
                    methods.append(method.replace('_', ' ').title())
                    remaining_games.append(pred['remaining_games'])
                    confidences.append(pred['confidence'])
            
            if methods:
                colors = ['green' if c == 'high' else 'orange' if c == 'medium' else 'red' for c in confidences]
                bars = axes[0, 1].bar(range(len(methods)), remaining_games, color=colors, alpha=0.7)
                axes[0, 1].set_xticks(range(len(methods)))
                axes[0, 1].set_xticklabels(methods, rotation=45, ha='right')
                axes[0, 1].set_title('Prediction Methods Comparison')
                axes[0, 1].set_ylabel('Remaining Games')
                
                # Add ensemble prediction line if available
                if 'ensemble' in prediction_results['predictions']:
                    ensemble_val = prediction_results['predictions']['ensemble']['remaining_games']
                    axes[0, 1].axhline(ensemble_val, color='purple', linestyle='--', linewidth=2, label=f'Ensemble: {ensemble_val}')
                    axes[0, 1].legend()
        
        # 3. Current streak in context
        if durations:
            percentile = stats.percentileofscore(durations, self.current_loss_streak_duration)
            
            # Create percentile visualization
            sorted_durations = sorted(durations)
            percentiles = [stats.percentileofscore(durations, d) for d in sorted_durations]
            
            axes[1, 0].plot(sorted_durations, percentiles, 'b-', linewidth=2)
            axes[1, 0].scatter([self.current_loss_streak_duration], [percentile], color='red', s=100, zorder=5, label=f'Current ({percentile:.1f}th percentile)')
            axes[1, 0].set_title('Current Streak Percentile Ranking')
            axes[1, 0].set_xlabel('Streak Duration')
            axes[1, 0].set_ylabel('Percentile')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Timeline of historical streaks
        if completed_streaks:
            # Plot timeline of loss streaks
            for i, streak in enumerate(completed_streaks):
                start_date = streak.start_date
                duration_days = (streak.end_date - streak.start_date).days
                
                # Color by transition type
                color = 'green' if 'win' in streak.transition_type else 'orange'
                axes[1, 1].barh(i, duration_days, left=start_date, height=0.8, color=color, alpha=0.7)
            
            axes[1, 1].set_title('Historical Loss Streaks Timeline')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Loss Streak #')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.7, label='→ Win Streak'),
                Patch(facecolor='orange', alpha=0.7, label='→ Neutral')
            ]
            axes[1, 1].legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/win_streak_prediction.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_prediction_report(self, prediction_results: Dict) -> str:
        """Generate comprehensive prediction report"""
        if 'error' in prediction_results:
            return f"Prediction Error: {prediction_results['error']}"
            
        current = prediction_results['current_status']
        historical = prediction_results['historical_context']
        predictions = prediction_results['predictions']
        
        report = f"""
# 🔮 Win Streak Transition Prediction Report

## Current Loss Streak Status
- **Duration**: {current['duration']} matches
- **Win Rate**: {current['win_rate']:.1%}
- **Historical Percentile**: {current['percentile']:.1f}th percentile

## Historical Context
- **Completed Loss Streaks Analyzed**: {historical['completed_streaks']}
- **Average Duration**: {historical['mean_duration']:.1f} matches
- **Median Duration**: {historical['median_duration']:.1f} matches
- **Duration Range**: {historical['duration_range']}
- **Standard Deviation**: {historical['std_duration']:.1f} matches

## 🎯 Predictions

"""
        
        # Sort predictions by remaining games for better readability
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1]['remaining_games'])
        
        for method, pred in sorted_predictions:
            confidence_emoji = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}[pred['confidence']]
            
            report += f"""### {method.replace('_', ' ').title()}
- **Remaining Games**: {pred['remaining_games']}
- **Confidence**: {confidence_emoji} {pred['confidence'].upper()}
- **Method**: {pred['method']}
"""
            
            if 'note' in pred:
                report += f"- **Note**: {pred['note']}\n"
            if 'trend_note' in pred:
                report += f"- **Trend**: {pred['trend_note']}\n"
                
            report += "\n"
        
        # Risk factors
        if prediction_results['risk_factors']:
            report += "## ⚠️ Risk Factors\n"
            for factor in prediction_results['risk_factors']:
                report += f"- {factor}\n"
            report += "\n"
        
        # Recommendation
        report += f"""## 💡 Recommendation
{prediction_results['recommendation']}

## 📊 Key Insights
- Your current loss streak of {current['duration']} matches is at the {current['percentile']:.1f}th percentile
- Historical loss streaks averaged {historical['mean_duration']:.1f} matches
- Most reliable prediction suggests **{predictions.get('ensemble', predictions.get('statistical_median', {})).get('remaining_games', 'N/A')} more games** until win streak

---
*Prediction generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report

def main():
    """Main function to run win streak prediction"""
    predictor = WinStreakPredictor("276939814")
    
    print("Analyzing historical loss streak patterns...")
    analysis_results = predictor.analyze_historical_patterns()
    
    if 'error' in analysis_results:
        print(f"Error: {analysis_results['error']}")
        return
        
    print(f"Found {analysis_results['total_loss_streaks']} historical loss streaks")
    print(f"Current loss streak duration: {analysis_results['current_duration']} matches")
    
    print("\nGenerating predictions...")
    prediction_results = predictor.predict_remaining_games()
    
    if 'error' in prediction_results:
        print(f"Prediction Error: {prediction_results['error']}")
        return
    
    print("Creating visualizations...")
    predictor.create_prediction_visualization(prediction_results)
    
    print("Generating report...")
    report = predictor.generate_prediction_report(prediction_results)
    
    with open('win_streak_prediction_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n" + "="*80)
    print(report)
    print("="*80)
    print("\nReport saved to win_streak_prediction_report.md")
    print("Visualizations saved to win_streak_prediction.png")

if __name__ == "__main__":
    main()
