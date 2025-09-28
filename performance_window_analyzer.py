#!/usr/bin/env python3
"""
Dota 2 Performance Window Analysis
Analyzes win/loss streaks as performance windows (20+ matches with specific win rates)
rather than consecutive wins/losses
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from dota_matchmaking_analysis import DotaMatchAnalyzer, MatchResult
from scipy import stats

@dataclass
class PerformanceWindow:
    """Represents a performance window (win/loss streak)"""
    start_index: int
    end_index: int
    match_count: int
    win_rate: float
    window_type: str  # 'win_streak', 'loss_streak', 'neutral'
    start_date: datetime
    end_date: datetime

class PerformanceWindowAnalyzer:
    """Analyzes performance windows in match history"""
    
    def __init__(self, player_id: str):
        self.player_id = player_id
        self.analyzer = DotaMatchAnalyzer(player_id)
        self.matches: List[MatchResult] = []
        self.performance_windows: List[PerformanceWindow] = []
        
        # Configurable thresholds
        self.window_size = 20  # Minimum matches for a performance window
        self.win_streak_threshold = 0.65  # 65%+ win rate for win streaks
        self.loss_streak_threshold = 0.35  # 35%- win rate for loss streaks
        
    def fetch_matches(self, days_back: int = 2000) -> None:
        """Fetch player matches"""
        self.analyzer.fetch_matches(days_back=days_back)
        self.matches = self.analyzer.matches
        print(f"Loaded {len(self.matches)} matches for performance window analysis")
        
    def identify_performance_windows(self) -> List[PerformanceWindow]:
        """Identify performance windows in match history"""
        if len(self.matches) < self.window_size:
            print(f"Need at least {self.window_size} matches for analysis")
            return []
            
        windows = []
        i = 0
        
        while i <= len(self.matches) - self.window_size:
            # Calculate rolling win rate for current window
            window_matches = self.matches[i:i + self.window_size]
            wins = sum(1 for match in window_matches if match.win)
            win_rate = wins / len(window_matches)
            
            # Determine window type
            if win_rate >= self.win_streak_threshold:
                window_type = 'win_streak'
            elif win_rate <= self.loss_streak_threshold:
                window_type = 'loss_streak'
            else:
                window_type = 'neutral'
                i += 1  # Move to next position for neutral windows
                continue
                
            # Extend the window as long as it maintains the same type
            end_index = i + self.window_size
            while end_index < len(self.matches):
                # Check if adding next match maintains the streak type
                extended_window = self.matches[i:end_index + 1]
                extended_wins = sum(1 for match in extended_window if match.win)
                extended_win_rate = extended_wins / len(extended_window)
                
                if window_type == 'win_streak' and extended_win_rate >= self.win_streak_threshold:
                    end_index += 1
                elif window_type == 'loss_streak' and extended_win_rate <= self.loss_streak_threshold:
                    end_index += 1
                else:
                    break
                    
            # Create performance window
            final_window = self.matches[i:end_index]
            final_wins = sum(1 for match in final_window if match.win)
            final_win_rate = final_wins / len(final_window)
            
            window = PerformanceWindow(
                start_index=i,
                end_index=end_index - 1,
                match_count=len(final_window),
                win_rate=final_win_rate,
                window_type=window_type,
                start_date=final_window[0].start_time,
                end_date=final_window[-1].start_time
            )
            
            windows.append(window)
            
            # Move to the end of current window
            i = end_index
            
        self.performance_windows = windows
        return windows
    
    def analyze_window_transitions(self) -> Dict:
        """Analyze transitions between performance windows"""
        if len(self.performance_windows) < 2:
            return {'error': 'Need at least 2 performance windows for transition analysis'}
            
        transitions = {
            'win_to_loss': 0,
            'loss_to_win': 0,
            'win_to_neutral': 0,
            'loss_to_neutral': 0,
            'neutral_to_win': 0,
            'neutral_to_loss': 0
        }
        
        win_to_loss_details = []
        loss_to_win_details = []
        
        for i in range(len(self.performance_windows) - 1):
            current = self.performance_windows[i]
            next_window = self.performance_windows[i + 1]
            
            transition_key = f"{current.window_type}_to_{next_window.window_type}"
            if transition_key in transitions:
                transitions[transition_key] += 1
                
            # Collect detailed information for key transitions
            if current.window_type == 'win_streak' and next_window.window_type == 'loss_streak':
                win_to_loss_details.append({
                    'win_streak_length': current.match_count,
                    'win_streak_rate': current.win_rate,
                    'loss_streak_length': next_window.match_count,
                    'loss_streak_rate': next_window.win_rate,
                    'transition_date': next_window.start_date
                })
                
            elif current.window_type == 'loss_streak' and next_window.window_type == 'win_streak':
                loss_to_win_details.append({
                    'loss_streak_length': current.match_count,
                    'loss_streak_rate': current.win_rate,
                    'win_streak_length': next_window.match_count,
                    'win_streak_rate': next_window.win_rate,
                    'transition_date': next_window.start_date
                })
        
        # Calculate probabilities
        total_win_streaks = sum(1 for w in self.performance_windows if w.window_type == 'win_streak')
        total_loss_streaks = sum(1 for w in self.performance_windows if w.window_type == 'loss_streak')
        
        prob_loss_after_win = transitions['win_to_loss'] / total_win_streaks if total_win_streaks > 0 else 0
        prob_win_after_loss = transitions['loss_to_win'] / total_loss_streaks if total_loss_streaks > 0 else 0
        
        # Expected probabilities (based on overall distribution)
        total_windows = len(self.performance_windows)
        expected_prob_loss = total_loss_streaks / total_windows if total_windows > 0 else 0
        expected_prob_win = total_win_streaks / total_windows if total_windows > 0 else 0
        
        return {
            'transitions': transitions,
            'win_to_loss_details': win_to_loss_details,
            'loss_to_win_details': loss_to_win_details,
            'probabilities': {
                'prob_loss_after_win': prob_loss_after_win,
                'prob_win_after_loss': prob_win_after_loss,
                'expected_prob_loss': expected_prob_loss,
                'expected_prob_win': expected_prob_win
            },
            'window_counts': {
                'total_win_streaks': total_win_streaks,
                'total_loss_streaks': total_loss_streaks,
                'total_windows': total_windows
            }
        }
    
    def test_forced_loss_theory(self) -> Dict:
        """Test if win streaks are systematically followed by loss streaks"""
        transition_analysis = self.analyze_window_transitions()
        
        if 'error' in transition_analysis:
            return transition_analysis
            
        probs = transition_analysis['probabilities']
        transitions = transition_analysis['transitions']
        win_to_loss_details = transition_analysis['win_to_loss_details']
        
        # Statistical tests
        statistical_tests = {}
        
        # Test 1: Are win streaks followed by loss streaks more than expected?
        if transitions['win_to_loss'] > 0 and transition_analysis['window_counts']['total_win_streaks'] > 0:
            # Binomial test
            n_win_streaks = transition_analysis['window_counts']['total_win_streaks']
            observed_transitions = transitions['win_to_loss']
            expected_prob = probs['expected_prob_loss']
            
            if expected_prob > 0:
                binomial_test = stats.binom_test(observed_transitions, n_win_streaks, expected_prob, alternative='greater')
                statistical_tests['binomial_test'] = {
                    'p_value': binomial_test,
                    'significant': binomial_test < 0.05
                }
        
        # Test 2: Do longer win streaks lead to longer loss streaks?
        if len(win_to_loss_details) > 3:
            win_lengths = [d['win_streak_length'] for d in win_to_loss_details]
            loss_lengths = [d['loss_streak_length'] for d in win_to_loss_details]
            
            correlation_test = stats.pearsonr(win_lengths, loss_lengths)
            statistical_tests['correlation_test'] = {
                'correlation': correlation_test[0],
                'p_value': correlation_test[1],
                'significant': correlation_test[1] < 0.05 and correlation_test[0] > 0.3
            }
        
        # Evidence assessment
        evidence_count = 0
        evidence_details = []
        
        # Check probability difference
        prob_diff = probs['prob_loss_after_win'] - probs['expected_prob_loss']
        if prob_diff > 0.2:  # 20% higher than expected
            evidence_count += 1
            evidence_details.append(f"Win streaks followed by loss streaks {probs['prob_loss_after_win']:.1%} of time (expected: {probs['expected_prob_loss']:.1%})")
        
        # Check statistical significance
        if 'binomial_test' in statistical_tests and statistical_tests['binomial_test']['significant']:
            evidence_count += 1
            evidence_details.append(f"Statistically significant bias (p={statistical_tests['binomial_test']['p_value']:.4f})")
        
        if 'correlation_test' in statistical_tests and statistical_tests['correlation_test']['significant']:
            evidence_count += 1
            evidence_details.append(f"Longer win streaks correlate with longer loss streaks (r={statistical_tests['correlation_test']['correlation']:.3f})")
        
        # Overall assessment
        if evidence_count >= 2:
            evidence_strength = 'STRONG'
            conclusion = "STRONG EVIDENCE FOR FORCED LOSSES: Multiple indicators suggest systematic bias in matchmaking."
        elif evidence_count >= 1:
            evidence_strength = 'MODERATE'
            conclusion = "MODERATE EVIDENCE FOR FORCED LOSSES: Some patterns suggest potential bias."
        else:
            evidence_strength = 'WEAK'
            conclusion = "WEAK EVIDENCE FOR FORCED LOSSES: Patterns appear consistent with natural variance."
        
        return {
            'transition_analysis': transition_analysis,
            'statistical_tests': statistical_tests,
            'evidence_details': evidence_details,
            'evidence_strength': evidence_strength,
            'conclusion': conclusion
        }
    
    def get_current_performance_status(self) -> Dict:
        """Analyze current performance status"""
        if len(self.matches) < self.window_size:
            return {'error': f'Need at least {self.window_size} matches for current status'}
            
        # Check last 20 matches
        recent_matches = self.matches[-self.window_size:]
        recent_wins = sum(1 for match in recent_matches if match.win)
        recent_win_rate = recent_wins / len(recent_matches)
        
        # Determine current status
        if recent_win_rate >= self.win_streak_threshold:
            current_status = 'WIN_STREAK'
            risk_level = 'HIGH'  # High risk of entering loss streak
            recommendation = "CAUTION: You're in a win streak. Based on forced loss theory, be prepared for potential downturn."
        elif recent_win_rate <= self.loss_streak_threshold:
            current_status = 'LOSS_STREAK'
            risk_level = 'MEDIUM'  # May continue or end soon
            recommendation = "PATIENCE: You're in a loss streak. This should end soon based on historical patterns."
        else:
            current_status = 'NEUTRAL'
            risk_level = 'LOW'
            recommendation = "STABLE: Normal performance window. Good time for consistent play."
        
        # Check trend in recent matches (last 10 vs previous 10)
        if len(self.matches) >= 40:
            last_10 = self.matches[-10:]
            prev_10 = self.matches[-20:-10]
            
            last_10_wr = sum(1 for m in last_10 if m.win) / 10
            prev_10_wr = sum(1 for m in prev_10 if m.win) / 10
            
            trend = "IMPROVING" if last_10_wr > prev_10_wr + 0.1 else "DECLINING" if last_10_wr < prev_10_wr - 0.1 else "STABLE"
        else:
            trend = "INSUFFICIENT_DATA"
        
        return {
            'current_status': current_status,
            'recent_win_rate': recent_win_rate,
            'recent_matches_analyzed': len(recent_matches),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'trend': trend,
            'matches_in_current_window': self._count_matches_in_current_window()
        }
    
    def _count_matches_in_current_window(self) -> int:
        """Count how many matches are in the current performance window"""
        if len(self.matches) < self.window_size:
            return len(self.matches)
            
        current_win_rate = sum(1 for m in self.matches[-self.window_size:] if m.win) / self.window_size
        
        if current_win_rate >= self.win_streak_threshold or current_win_rate <= self.loss_streak_threshold:
            # Count backwards to find start of current window
            window_count = self.window_size
            
            for i in range(len(self.matches) - self.window_size - 1, -1, -1):
                test_window = self.matches[i:len(self.matches)]
                test_win_rate = sum(1 for m in test_window if m.win) / len(test_window)
                
                if current_win_rate >= self.win_streak_threshold:
                    if test_win_rate >= self.win_streak_threshold:
                        window_count = len(test_window)
                    else:
                        break
                elif current_win_rate <= self.loss_streak_threshold:
                    if test_win_rate <= self.loss_streak_threshold:
                        window_count = len(test_window)
                    else:
                        break
                        
            return window_count
        else:
            return 0  # Not in a performance window
    
    def create_visualizations(self, output_dir: str = ".") -> None:
        """Create visualizations of performance windows"""
        if not self.performance_windows:
            print("No performance windows to visualize")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dota 2 Performance Window Analysis', fontsize=16)
        
        # 1. Performance windows timeline
        dates = [match.start_time for match in self.matches]
        win_rates = []
        
        # Calculate rolling win rate
        for i in range(self.window_size - 1, len(self.matches)):
            window = self.matches[i - self.window_size + 1:i + 1]
            wr = sum(1 for m in window if m.win) / len(window)
            win_rates.append(wr)
            
        timeline_dates = dates[self.window_size - 1:]
        
        axes[0, 0].plot(timeline_dates, win_rates, linewidth=1, alpha=0.7)
        axes[0, 0].axhline(y=self.win_streak_threshold, color='green', linestyle='--', alpha=0.7, label=f'Win Streak ({self.win_streak_threshold:.0%})')
        axes[0, 0].axhline(y=self.loss_streak_threshold, color='red', linestyle='--', alpha=0.7, label=f'Loss Streak ({self.loss_streak_threshold:.0%})')
        axes[0, 0].axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, label='50%')
        
        # Highlight performance windows
        for window in self.performance_windows:
            start_date = window.start_date
            end_date = window.end_date
            color = 'green' if window.window_type == 'win_streak' else 'red'
            axes[0, 0].axvspan(start_date, end_date, alpha=0.2, color=color)
            
        axes[0, 0].set_title(f'Rolling Win Rate ({self.window_size} games)')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Performance window distribution
        win_streaks = [w for w in self.performance_windows if w.window_type == 'win_streak']
        loss_streaks = [w for w in self.performance_windows if w.window_type == 'loss_streak']
        
        win_lengths = [w.match_count for w in win_streaks]
        loss_lengths = [w.match_count for w in loss_streaks]
        
        if win_lengths and loss_lengths:
            axes[0, 1].hist([win_lengths, loss_lengths], bins=10, alpha=0.7, 
                           label=['Win Streaks', 'Loss Streaks'], color=['green', 'red'])
            axes[0, 1].set_title('Performance Window Length Distribution')
            axes[0, 1].set_xlabel('Window Length (matches)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
        
        # 3. Win rate distribution in windows
        win_rates_in_wins = [w.win_rate for w in win_streaks]
        win_rates_in_losses = [w.win_rate for w in loss_streaks]
        
        if win_rates_in_wins and win_rates_in_losses:
            axes[1, 0].hist([win_rates_in_wins, win_rates_in_losses], bins=10, alpha=0.7,
                           label=['Win Streaks', 'Loss Streaks'], color=['green', 'red'])
            axes[1, 0].set_title('Win Rate Distribution in Performance Windows')
            axes[1, 0].set_xlabel('Win Rate')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
        
        # 4. Transition analysis
        transition_analysis = self.analyze_window_transitions()
        if 'transitions' in transition_analysis:
            transitions = transition_analysis['transitions']
            
            # Create transition matrix
            labels = ['Win Streak', 'Loss Streak', 'Neutral']
            matrix = np.array([
                [0, transitions['win_to_loss'], transitions['win_to_neutral']],
                [transitions['loss_to_win'], 0, transitions['loss_to_neutral']],
                [transitions['neutral_to_win'], transitions['neutral_to_loss'], 0]
            ])
            
            sns.heatmap(matrix, annot=True, fmt='d', xticklabels=labels, yticklabels=labels,
                       ax=axes[1, 1], cmap='Blues')
            axes[1, 1].set_title('Performance Window Transitions')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_windows_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self) -> str:
        """Generate comprehensive performance window analysis report"""
        if not self.performance_windows:
            return "No performance windows identified. Need more matches or adjust thresholds."
            
        forced_loss_analysis = self.test_forced_loss_theory()
        current_status = self.get_current_performance_status()
        
        win_streaks = [w for w in self.performance_windows if w.window_type == 'win_streak']
        loss_streaks = [w for w in self.performance_windows if w.window_type == 'loss_streak']
        
        report = f"""
# Dota 2 Performance Window Analysis Report

## Overview
- **Player ID**: {self.player_id}
- **Analysis Period**: {self.matches[0].start_time.strftime('%Y-%m-%d')} to {self.matches[-1].start_time.strftime('%Y-%m-%d')}
- **Total Matches**: {len(self.matches)}
- **Window Size**: {self.window_size} matches
- **Win Streak Threshold**: {self.win_streak_threshold:.0%}
- **Loss Streak Threshold**: {self.loss_streak_threshold:.0%}

## Performance Windows Identified
- **Total Performance Windows**: {len(self.performance_windows)}
- **Win Streaks**: {len(win_streaks)} (avg length: {np.mean([w.match_count for w in win_streaks]):.1f} matches)
- **Loss Streaks**: {len(loss_streaks)} (avg length: {np.mean([w.match_count for w in loss_streaks]):.1f} matches)

## Current Status
- **Current Performance**: {current_status.get('current_status', 'Unknown')}
- **Recent Win Rate ({self.window_size} games)**: {current_status.get('recent_win_rate', 0):.1%}
- **Risk Level**: {current_status.get('risk_level', 'Unknown')}
- **Matches in Current Window**: {current_status.get('matches_in_current_window', 0)}

## Forced Loss Theory Analysis
**{forced_loss_analysis.get('conclusion', 'Analysis incomplete')}**

**Evidence Strength**: {forced_loss_analysis.get('evidence_strength', 'Unknown')}

### Key Findings:
"""
        
        for detail in forced_loss_analysis.get('evidence_details', []):
            report += f"- {detail}\n"
            
        if 'transition_analysis' in forced_loss_analysis:
            trans_analysis = forced_loss_analysis['transition_analysis']
            probs = trans_analysis['probabilities']
            
            report += f"""
### Transition Probabilities
- **Win Streak → Loss Streak**: {probs['prob_loss_after_win']:.1%}
- **Expected Probability**: {probs['expected_prob_loss']:.1%}
- **Difference**: {(probs['prob_loss_after_win'] - probs['expected_prob_loss']):.1%}

## Recommendation
{current_status.get('recommendation', 'Continue monitoring performance.')}

---
*Analysis generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report

def main():
    """Main function to run performance window analysis"""
    analyzer = PerformanceWindowAnalyzer("276939814")
    
    print("Fetching match data...")
    analyzer.fetch_matches(days_back=2000)
    
    print("Identifying performance windows...")
    windows = analyzer.identify_performance_windows()
    
    if not windows:
        print("No performance windows found. Try adjusting thresholds or getting more match data.")
        return
        
    print(f"Found {len(windows)} performance windows")
    
    print("Analyzing forced loss theory...")
    forced_loss_results = analyzer.test_forced_loss_theory()
    
    print("Getting current status...")
    current_status = analyzer.get_current_performance_status()
    
    print("Creating visualizations...")
    analyzer.create_visualizations()
    
    print("Generating report...")
    report = analyzer.generate_report()
    
    with open('performance_windows_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
        
    print("\n" + "="*60)
    print(report)
    print("="*60)
    print("\nReport saved to performance_windows_report.md")
    print("Visualizations saved to performance_windows_analysis.png")

if __name__ == "__main__":
    main()
