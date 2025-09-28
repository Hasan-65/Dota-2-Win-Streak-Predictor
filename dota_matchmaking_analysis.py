#!/usr/bin/env python3
"""
Dota 2 Matchmaking Analysis Script
Analyzes player match history to detect patterns in win/loss streaks
and test the theory of forced losses after winning streaks.
"""

import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from scipy import stats
import argparse
from dataclasses import dataclass
import time

@dataclass
class MatchResult:
    """Represents a single match result"""
    match_id: int
    win: bool
    start_time: datetime
    duration: int
    hero_id: int
    kills: int
    deaths: int
    assists: int
    game_mode: int
    
class DotaMatchAnalyzer:
    """Main class for analyzing Dota 2 match patterns"""
    
    def __init__(self, player_id: str, api_key: Optional[str] = None):
        self.player_id = player_id
        self.api_key = api_key
        self.base_url = "https://api.opendota.com/api"
        self.matches: List[MatchResult] = []
        self.session = requests.Session()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make rate-limited API request"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
            
        if params is None:
            params = {}
            
        if self.api_key:
            params['api_key'] = self.api_key
            
        try:
            response = self.session.get(f"{self.base_url}/{endpoint}", params=params)
            response.raise_for_status()
            self.last_request_time = time.time()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return {}
    
    def fetch_matches(self, days_back: int = 365, limit: int = None) -> None:
        """Fetch player matches from the last N days"""
        print(f"Fetching matches for player {self.player_id} from last {days_back} days...")
        
        # Calculate date threshold
        cutoff_date = datetime.now() - timedelta(days=days_back)
        cutoff_timestamp = int(cutoff_date.timestamp())
        
        # Fetch all matches in batches (OpenDota API has pagination)
        all_matches = []
        offset = 0
        batch_size = 100  # Fetch in batches of 100
        
        while True:
            params = {'limit': batch_size, 'offset': offset}
            if limit and len(all_matches) >= limit:
                break
                
            print(f"Fetching batch {offset//batch_size + 1} (offset: {offset})...")
            matches_batch = self._make_request(f"players/{self.player_id}/matches", params)
            
            if not matches_batch or len(matches_batch) == 0:
                break
                
            all_matches.extend(matches_batch)
            
            # If we got fewer matches than batch_size, we've reached the end
            if len(matches_batch) < batch_size:
                break
                
            offset += batch_size
            
            # Safety limit to prevent infinite loops
            if offset > 10000:  # Max 10k matches
                print("Reached safety limit of 10,000 matches")
                break
        
        matches_data = all_matches
        if not matches_data:
            print("Failed to fetch matches")
            return
            
        print(f"Retrieved {len(matches_data)} total matches")
        
        # Filter matches by date and convert to MatchResult objects
        filtered_matches = []
        for match in matches_data:
            match_time = datetime.fromtimestamp(match['start_time'])
            
            if match_time < cutoff_date:
                continue
                
            # Determine if player won (radiant_win and player_slot < 128 means radiant player won)
            # or (not radiant_win and player_slot >= 128 means dire player won)
            player_slot = match.get('player_slot', 0)
            radiant_win = match.get('radiant_win', False)
            
            if player_slot < 128:  # Radiant player
                won = radiant_win
            else:  # Dire player
                won = not radiant_win
                
            match_result = MatchResult(
                match_id=match['match_id'],
                win=won,
                start_time=match_time,
                duration=match.get('duration', 0),
                hero_id=match.get('hero_id', 0),
                kills=match.get('kills', 0),
                deaths=match.get('deaths', 0),
                assists=match.get('assists', 0),
                game_mode=match.get('game_mode', 0)
            )
            filtered_matches.append(match_result)
            
        # Sort by start time (oldest first)
        self.matches = sorted(filtered_matches, key=lambda x: x.start_time)
        print(f"Analyzed {len(self.matches)} matches within the date range")
        
    def analyze_streaks(self) -> Dict:
        """Analyze win/loss streaks in match history"""
        if not self.matches:
            return {}
            
        results = [match.win for match in self.matches]
        
        # Find streaks
        streaks = []
        current_streak = 1
        current_type = results[0]  # True for win streak, False for loss streak
        
        for i in range(1, len(results)):
            if results[i] == current_type:
                current_streak += 1
            else:
                streaks.append((current_type, current_streak))
                current_streak = 1
                current_type = results[i]
                
        # Add the last streak
        streaks.append((current_type, current_streak))
        
        # Separate win and loss streaks
        win_streaks = [length for is_win, length in streaks if is_win]
        loss_streaks = [length for is_win, length in streaks if not is_win]
        
        # Calculate statistics
        total_wins = sum(results)
        total_losses = len(results) - total_wins
        win_rate = total_wins / len(results) if results else 0
        
        analysis = {
            'total_matches': len(results),
            'total_wins': total_wins,
            'total_losses': total_losses,
            'win_rate': win_rate,
            'win_streaks': win_streaks,
            'loss_streaks': loss_streaks,
            'avg_win_streak': np.mean(win_streaks) if win_streaks else 0,
            'avg_loss_streak': np.mean(loss_streaks) if loss_streaks else 0,
            'max_win_streak': max(win_streaks) if win_streaks else 0,
            'max_loss_streak': max(loss_streaks) if loss_streaks else 0,
            'streak_sequence': streaks
        }
        
        return analysis
        
    def test_forced_loss_theory(self) -> Dict:
        """Test if loss streaks are more likely to follow win streaks (streak-to-streak analysis)"""
        if not self.matches:
            return {}
            
        streak_analysis = self.analyze_streaks()
        streaks = streak_analysis['streak_sequence']
        
        if len(streaks) < 2:
            return {'error': 'Not enough streaks to analyze'}
            
        # Focus on meaningful streaks (2+ games) for the forced loss theory
        meaningful_streaks = [(is_win, length) for is_win, length in streaks if length >= 2]
        
        if len(meaningful_streaks) < 2:
            return {'error': 'Not enough meaningful streaks (2+ games) to analyze'}
            
        # Analyze streak-to-streak transitions (only meaningful streaks)
        win_streak_to_loss_streak = 0
        loss_streak_to_win_streak = 0
        win_streak_to_win_streak = 0
        loss_streak_to_loss_streak = 0
        
        # Track lengths of streaks that follow win streaks vs all streaks
        loss_streaks_after_win_streaks = []
        win_streaks_after_loss_streaks = []
        loss_streaks_after_loss_streaks = []
        win_streaks_after_win_streaks = []
        
        # Also analyze by win streak length (longer win streaks should lead to longer loss streaks if theory is true)
        loss_after_short_wins = []  # Loss streaks after 2-3 win games
        loss_after_medium_wins = []  # Loss streaks after 4-6 win games  
        loss_after_long_wins = []   # Loss streaks after 7+ win games
        
        for i in range(len(meaningful_streaks) - 1):
            current_type, current_length = meaningful_streaks[i]
            next_type, next_length = meaningful_streaks[i + 1]
            
            if current_type and not next_type:  # Win streak → Loss streak
                win_streak_to_loss_streak += 1
                loss_streaks_after_win_streaks.append(next_length)
                
                # Categorize by win streak length
                if current_length <= 3:
                    loss_after_short_wins.append(next_length)
                elif current_length <= 6:
                    loss_after_medium_wins.append(next_length)
                else:
                    loss_after_long_wins.append(next_length)
                    
            elif not current_type and next_type:  # Loss streak → Win streak
                loss_streak_to_win_streak += 1
                win_streaks_after_loss_streaks.append(next_length)
                
            elif current_type and next_type:  # Win streak → Win streak
                win_streak_to_win_streak += 1
                win_streaks_after_win_streaks.append(next_length)
                
            else:  # Loss streak → Loss streak
                loss_streak_to_loss_streak += 1
                loss_streaks_after_loss_streaks.append(next_length)
        
        # Calculate probabilities for meaningful streaks
        total_meaningful_win_streaks = win_streak_to_loss_streak + win_streak_to_win_streak
        total_meaningful_loss_streaks = loss_streak_to_win_streak + loss_streak_to_loss_streak
        
        prob_loss_streak_after_win_streak = win_streak_to_loss_streak / total_meaningful_win_streaks if total_meaningful_win_streaks > 0 else 0
        prob_win_streak_after_loss_streak = loss_streak_to_win_streak / total_meaningful_loss_streaks if total_meaningful_loss_streaks > 0 else 0
        
        # Expected probability based on overall streak distribution
        total_meaningful_streaks = len(meaningful_streaks)
        meaningful_win_streaks = len([s for s in meaningful_streaks if s[0]])
        meaningful_loss_streaks = len([s for s in meaningful_streaks if not s[0]])
        
        expected_prob_loss_streak = meaningful_loss_streaks / total_meaningful_streaks if total_meaningful_streaks > 0 else 0
        expected_prob_win_streak = meaningful_win_streaks / total_meaningful_streaks if total_meaningful_streaks > 0 else 0
        
        # Statistical tests
        all_loss_streaks = [length for is_win, length in meaningful_streaks if not is_win]
        all_win_streaks = [length for is_win, length in meaningful_streaks if is_win]
        
        # Test 1: Are loss streaks after win streaks longer than all loss streaks?
        forced_loss_length_test = None
        if loss_streaks_after_win_streaks and all_loss_streaks:
            forced_loss_length_test = stats.mannwhitneyu(
                loss_streaks_after_win_streaks, 
                all_loss_streaks, 
                alternative='greater'
            )
        
        # Test 2: Do longer win streaks lead to longer loss streaks?
        win_streak_correlation_test = None
        if len(loss_streaks_after_win_streaks) > 5:  # Need reasonable sample size
            # Get corresponding win streak lengths
            win_lengths_before_loss = []
            for i in range(len(meaningful_streaks) - 1):
                current_type, current_length = meaningful_streaks[i]
                next_type, next_length = meaningful_streaks[i + 1]
                if current_type and not next_type:
                    win_lengths_before_loss.append(current_length)
            
            if len(win_lengths_before_loss) == len(loss_streaks_after_win_streaks):
                win_streak_correlation_test = stats.pearsonr(win_lengths_before_loss, loss_streaks_after_win_streaks)
        
        # Test 3: Chi-square test for independence of streak transitions
        transition_independence_test = None
        if total_meaningful_win_streaks > 0 and total_meaningful_loss_streaks > 0:
            # Create contingency table: [win_to_loss, win_to_win], [loss_to_win, loss_to_loss]
            contingency_table = [
                [win_streak_to_loss_streak, win_streak_to_win_streak],
                [loss_streak_to_win_streak, loss_streak_to_loss_streak]
            ]
            if all(sum(row) > 0 for row in contingency_table):
                transition_independence_test = stats.chi2_contingency(contingency_table)
            
        return {
            'meaningful_streaks_analysis': {
                'total_meaningful_streaks': len(meaningful_streaks),
                'meaningful_win_streaks': meaningful_win_streaks,
                'meaningful_loss_streaks': meaningful_loss_streaks,
                'min_streak_length': 2
            },
            'transition_analysis': {
                'win_streak_to_loss_streak': win_streak_to_loss_streak,
                'loss_streak_to_win_streak': loss_streak_to_win_streak,
                'prob_loss_streak_after_win_streak': prob_loss_streak_after_win_streak,
                'prob_win_streak_after_loss_streak': prob_win_streak_after_loss_streak,
                'expected_prob_loss_streak': expected_prob_loss_streak,
                'expected_prob_win_streak': expected_prob_win_streak
            },
            'streak_length_analysis': {
                'avg_loss_streak_after_win_streak': np.mean(loss_streaks_after_win_streaks) if loss_streaks_after_win_streaks else 0,
                'avg_win_streak_after_loss_streak': np.mean(win_streaks_after_loss_streaks) if win_streaks_after_loss_streaks else 0,
                'avg_all_meaningful_loss_streaks': np.mean(all_loss_streaks) if all_loss_streaks else 0,
                'avg_all_meaningful_win_streaks': np.mean(all_win_streaks) if all_win_streaks else 0,
                'loss_after_short_wins': np.mean(loss_after_short_wins) if loss_after_short_wins else 0,
                'loss_after_medium_wins': np.mean(loss_after_medium_wins) if loss_after_medium_wins else 0,
                'loss_after_long_wins': np.mean(loss_after_long_wins) if loss_after_long_wins else 0
            },
            'statistical_tests': {
                'forced_loss_length_test': forced_loss_length_test,
                'win_streak_correlation_test': win_streak_correlation_test,
                'transition_independence_test': transition_independence_test,
                'interpretations': self._interpret_forced_loss_tests(
                    forced_loss_length_test, 
                    win_streak_correlation_test, 
                    transition_independence_test,
                    prob_loss_streak_after_win_streak,
                    expected_prob_loss_streak
                )
            }
        }
        
    def _interpret_forced_loss_tests(self, length_test, correlation_test, independence_test, 
                                   prob_loss_after_win, expected_prob_loss) -> Dict:
        """Interpret the multiple statistical test results for forced loss theory"""
        interpretations = {
            'overall_conclusion': '',
            'detailed_findings': [],
            'evidence_strength': 'none'
        }
        
        evidence_count = 0
        
        # Test 1: Length comparison
        if length_test:
            if length_test.pvalue < 0.05:
                interpretations['detailed_findings'].append(
                    f"✓ SIGNIFICANT: Loss streaks after win streaks are longer than average (p={length_test.pvalue:.4f})"
                )
                evidence_count += 1
            else:
                interpretations['detailed_findings'].append(
                    f"✗ Loss streak lengths after wins are not significantly different (p={length_test.pvalue:.4f})"
                )
        
        # Test 2: Correlation between win streak length and following loss streak length
        if correlation_test:
            correlation, p_value = correlation_test
            if p_value < 0.05 and correlation > 0.3:
                interpretations['detailed_findings'].append(
                    f"✓ SIGNIFICANT: Longer win streaks correlate with longer loss streaks (r={correlation:.3f}, p={p_value:.4f})"
                )
                evidence_count += 1
            elif p_value < 0.05 and correlation < -0.3:
                interpretations['detailed_findings'].append(
                    f"✗ Longer win streaks actually correlate with shorter loss streaks (r={correlation:.3f}, p={p_value:.4f})"
                )
            else:
                interpretations['detailed_findings'].append(
                    f"✗ No significant correlation between win streak and loss streak lengths (r={correlation:.3f}, p={p_value:.4f})"
                )
        
        # Test 3: Independence of transitions
        if independence_test:
            chi2, p_value, dof, expected = independence_test
            if p_value < 0.05:
                interpretations['detailed_findings'].append(
                    f"✓ SIGNIFICANT: Streak transitions are not independent (χ²={chi2:.3f}, p={p_value:.4f})"
                )
                evidence_count += 1
            else:
                interpretations['detailed_findings'].append(
                    f"✗ Streak transitions appear independent/random (χ²={chi2:.3f}, p={p_value:.4f})"
                )
        
        # Probability analysis
        prob_difference = prob_loss_after_win - expected_prob_loss
        if abs(prob_difference) > 0.1:  # More than 10% difference
            if prob_difference > 0:
                interpretations['detailed_findings'].append(
                    f"✓ Win streaks are followed by loss streaks {prob_loss_after_win:.1%} of the time (expected: {expected_prob_loss:.1%})"
                )
                evidence_count += 0.5  # Weaker evidence
            else:
                interpretations['detailed_findings'].append(
                    f"✗ Win streaks are actually less likely to be followed by loss streaks than expected"
                )
        else:
            interpretations['detailed_findings'].append(
                f"✗ Win→Loss transition probability is close to expected ({prob_loss_after_win:.1%} vs {expected_prob_loss:.1%})"
            )
        
        # Overall conclusion
        if evidence_count >= 2:
            interpretations['evidence_strength'] = 'strong'
            interpretations['overall_conclusion'] = "STRONG EVIDENCE FOR FORCED LOSSES: Multiple statistical tests support the theory that the matchmaking system forces loss streaks after win streaks."
        elif evidence_count >= 1:
            interpretations['evidence_strength'] = 'moderate' 
            interpretations['overall_conclusion'] = "MODERATE EVIDENCE FOR FORCED LOSSES: Some statistical evidence supports the theory, but it's not conclusive."
        elif evidence_count >= 0.5:
            interpretations['evidence_strength'] = 'weak'
            interpretations['overall_conclusion'] = "WEAK EVIDENCE FOR FORCED LOSSES: Minor patterns detected, but could be due to natural variance."
        else:
            interpretations['evidence_strength'] = 'none'
            interpretations['overall_conclusion'] = "NO EVIDENCE FOR FORCED LOSSES: Statistical analysis shows patterns consistent with fair matchmaking."
            
        return interpretations
            
    def create_visualizations(self, output_dir: str = ".") -> None:
        """Create visualizations of match patterns"""
        if not self.matches:
            print("No matches to visualize")
            return
            
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dota 2 Matchmaking Analysis', fontsize=16)
        
        # 1. Win/Loss timeline
        dates = [match.start_time for match in self.matches]
        results = [1 if match.win else 0 for match in self.matches]
        
        axes[0, 0].scatter(dates, results, alpha=0.6, c=['red' if r == 0 else 'green' for r in results])
        axes[0, 0].set_title('Win/Loss Timeline')
        axes[0, 0].set_ylabel('Result (1=Win, 0=Loss)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Streak length distribution
        streak_analysis = self.analyze_streaks()
        win_streaks = streak_analysis['win_streaks']
        loss_streaks = streak_analysis['loss_streaks']
        
        all_streaks = win_streaks + loss_streaks
        streak_types = ['Win'] * len(win_streaks) + ['Loss'] * len(loss_streaks)
        
        if all_streaks:
            axes[0, 1].hist([win_streaks, loss_streaks], bins=range(1, max(all_streaks) + 2), 
                           alpha=0.7, label=['Win Streaks', 'Loss Streaks'], color=['green', 'red'])
            axes[0, 1].set_title('Streak Length Distribution')
            axes[0, 1].set_xlabel('Streak Length')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
        
        # 3. Rolling win rate
        window_size = min(20, len(self.matches) // 4)
        if window_size > 0:
            rolling_wins = pd.Series(results).rolling(window=window_size, min_periods=1).mean()
            axes[1, 0].plot(dates, rolling_wins, linewidth=2)
            axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Win Rate')
            axes[1, 0].set_title(f'Rolling Win Rate (Window: {window_size} games)')
            axes[1, 0].set_ylabel('Win Rate')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].legend()
        
        # 4. Streak transition analysis
        forced_loss_analysis = self.test_forced_loss_theory()
        if 'meaningful_streaks_analysis' in forced_loss_analysis:
            trans = forced_loss_analysis['transition_analysis']
            length_analysis = forced_loss_analysis['streak_length_analysis']
            
            # Create bar chart comparing streak lengths
            categories = ['All Loss Streaks', 'Loss After Win Streak', 'Loss After Short Wins', 'Loss After Long Wins']
            values = [
                length_analysis['avg_all_meaningful_loss_streaks'],
                length_analysis['avg_loss_streak_after_win_streak'],
                length_analysis['loss_after_short_wins'] if length_analysis['loss_after_short_wins'] > 0 else 0,
                length_analysis['loss_after_long_wins'] if length_analysis['loss_after_long_wins'] > 0 else 0
            ]
            
            colors = ['blue', 'red', 'orange', 'darkred']
            bars = axes[1, 1].bar(range(len(categories)), values, color=colors, alpha=0.7)
            axes[1, 1].set_xticks(range(len(categories)))
            axes[1, 1].set_xticklabels(categories, rotation=45, ha='right')
            axes[1, 1].set_ylabel('Average Streak Length')
            axes[1, 1].set_title('Loss Streak Length Comparison')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if value > 0:
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                   f'{value:.1f}', ha='center', va='bottom')
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor streak analysis', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/dota_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report"""
        if not self.matches:
            return "No matches found for analysis."
            
        streak_analysis = self.analyze_streaks()
        forced_loss_analysis = self.test_forced_loss_theory()
        
        report = f"""
# Dota 2 Matchmaking Analysis Report

## Overview
- **Player ID**: {self.player_id}
- **Analysis Period**: {self.matches[0].start_time.strftime('%Y-%m-%d')} to {self.matches[-1].start_time.strftime('%Y-%m-%d')}
- **Total Matches**: {streak_analysis['total_matches']}

## Basic Statistics
- **Total Wins**: {streak_analysis['total_wins']}
- **Total Losses**: {streak_analysis['total_losses']}
- **Win Rate**: {streak_analysis['win_rate']:.2%}

## Streak Analysis
- **Average Win Streak**: {streak_analysis['avg_win_streak']:.1f} games
- **Average Loss Streak**: {streak_analysis['avg_loss_streak']:.1f} games
- **Maximum Win Streak**: {streak_analysis['max_win_streak']} games
- **Maximum Loss Streak**: {streak_analysis['max_loss_streak']} games
- **Total Win Streaks**: {len(streak_analysis['win_streaks'])}
- **Total Loss Streaks**: {len(streak_analysis['loss_streaks'])}

## Forced Loss Theory Analysis (Streak-to-Streak)
*Analysis focuses on meaningful streaks (2+ consecutive games) to test if win streaks are systematically followed by loss streaks.*
"""
        
        if 'meaningful_streaks_analysis' in forced_loss_analysis:
            meaningful = forced_loss_analysis['meaningful_streaks_analysis']
            trans = forced_loss_analysis['transition_analysis']
            length_analysis = forced_loss_analysis['streak_length_analysis']
            stat_tests = forced_loss_analysis['statistical_tests']
            interpretations = stat_tests['interpretations']
            
            report += f"""
### Meaningful Streaks Overview
- **Total Meaningful Streaks**: {meaningful['total_meaningful_streaks']} (minimum {meaningful['min_streak_length']} games each)
- **Win Streaks (2+ games)**: {meaningful['meaningful_win_streaks']}
- **Loss Streaks (2+ games)**: {meaningful['meaningful_loss_streaks']}

### Streak-to-Streak Transitions
- **Win Streaks → Loss Streaks**: {trans['win_streak_to_loss_streak']} times
- **Loss Streaks → Win Streaks**: {trans['loss_streak_to_win_streak']} times
- **Probability of Loss Streak After Win Streak**: {trans['prob_loss_streak_after_win_streak']:.1%}
- **Expected Probability**: {trans['expected_prob_loss_streak']:.1%}

### Streak Length Analysis
- **Average Loss Streak After Win Streak**: {length_analysis['avg_loss_streak_after_win_streak']:.1f} games
- **Average All Loss Streaks**: {length_analysis['avg_all_meaningful_loss_streaks']:.1f} games
- **Loss Streaks After Short Wins (2-3)**: {length_analysis['loss_after_short_wins']:.1f} games
- **Loss Streaks After Medium Wins (4-6)**: {length_analysis['loss_after_medium_wins']:.1f} games
- **Loss Streaks After Long Wins (7+)**: {length_analysis['loss_after_long_wins']:.1f} games

### Statistical Test Results
"""
            
            for finding in interpretations['detailed_findings']:
                report += f"- {finding}\n"
                
            report += f"""
### Overall Assessment
**{interpretations['overall_conclusion']}**

*Evidence Strength: {interpretations['evidence_strength'].upper()}*

## Conclusion
"""
        
        else:
            report += "\nInsufficient data for forced loss analysis."
            
        report += f"""

## Recommendations
1. **Sample Size**: Analyze more matches if possible for stronger statistical power
2. **External Factors**: Consider factors like time of day, patch changes, and team composition
3. **Skill Rating**: Monitor your MMR changes alongside win/loss patterns
4. **Psychological Factors**: Be aware that perceived patterns might be due to confirmation bias

---
*Analysis generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report

def main():
    """Main function to run the analysis"""
    # Hardcoded player ID for Mars (276939814)
    PLAYER_ID = "276939814"
    
    parser = argparse.ArgumentParser(description='Analyze Dota 2 matchmaking patterns')
    parser.add_argument('--player-id', default=PLAYER_ID, help=f'Your Dota 2 player ID (default: {PLAYER_ID})')
    parser.add_argument('--days', type=int, default=365, help='Number of days to analyze (default: 365)')
    parser.add_argument('--api-key', help='OpenDota API key (optional, for higher rate limits)')
    parser.add_argument('--output', default='.', help='Output directory for reports and visualizations')
    parser.add_argument('--limit', type=int, help='Limit number of matches to analyze')
    
    args = parser.parse_args()
    
    # Initialize analyzer with hardcoded or provided player ID
    analyzer = DotaMatchAnalyzer(args.player_id, args.api_key)
    
    # Fetch and analyze matches
    print("Starting Dota 2 matchmaking analysis...")
    analyzer.fetch_matches(days_back=args.days, limit=args.limit)
    
    if not analyzer.matches:
        print("No matches found. Please check your player ID and try again.")
        return
        
    # Generate analysis
    print("Analyzing match patterns...")
    
    # Create visualizations
    print("Creating visualizations...")
    analyzer.create_visualizations(args.output)
    
    # Generate and save report
    print("Generating report...")
    report = analyzer.generate_report()
    
    with open(f'{args.output}/dota_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"\nAnalysis complete! Check {args.output}/ for:")
    print("- dota_analysis_report.md (detailed report)")
    print("- dota_analysis.png (visualizations)")
    print("\nReport summary:")
    print("=" * 50)
    print(report)

if __name__ == "__main__":
    main()
