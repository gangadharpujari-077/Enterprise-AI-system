"""
Generative AI Module for Report Generation
Generate AI-based management reports using LLM APIs
"""

from typing import Dict, Optional, List
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenAIReportGenerator:
    """Generate intelligent AI-based management reports"""
    
    def __init__(self, api_provider: str = 'openai'):
        """
        Initialize GenAI report generator
        
        Args:
            api_provider: 'openai' or other LLM provider
        """
        self.api_provider = api_provider
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LLM API client"""
        if self.api_provider == 'openai':
            try:
                from openai import OpenAI
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    logger.warning("OPENAI_API_KEY not set. Report generation will be limited.")
                    return None
                return OpenAI(api_key=api_key)
            except ImportError:
                logger.warning("OpenAI package not available. Install: pip install openai")
                return None
        return None
    
    def generate_risk_report(self, employee_name: str,
                            predictions: Dict,
                            cluster_info: Dict,
                            metrics: Dict) -> str:
        """
        Generate risk analysis report for an employee
        
        Args:
            employee_name: Name of employee
            predictions: Dictionary with delay_risk, burnout_risk, performance_score
            cluster_info: Cluster assignment and characteristics
            metrics: Current work metrics
        
        Returns:
            Generated report text
        """
        prompt = self._build_risk_report_prompt(
            employee_name, predictions, cluster_info, metrics
        )
        
        return self._call_llm(prompt)
    
    def generate_team_report(self, team_name: str,
                           team_metrics: Dict,
                           cluster_distribution: Dict,
                           risk_summary: Dict) -> str:
        """
        Generate team-level report
        
        Args:
            team_name: Name of team
            team_metrics: Aggregated team metrics
            cluster_distribution: Distribution of employees across clusters
            risk_summary: Summary of risks in the team
        
        Returns:
            Generated team report
        """
        prompt = self._build_team_report_prompt(
            team_name, team_metrics, cluster_distribution, risk_summary
        )
        
        return self._call_llm(prompt)
    
    def generate_organizational_report(self, organization_metrics: Dict,
                                       risk_trends: Dict,
                                       recommendations: List[str]) -> str:
        """
        Generate organization-wide report
        
        Args:
            organization_metrics: Organization-level metrics
            risk_trends: Current risk trends
            recommendations: Strategic recommendations
        
        Returns:
            Generated organizational report
        """
        prompt = self._build_organizational_report_prompt(
            organization_metrics, risk_trends, recommendations
        )
        
        return self._call_llm(prompt)
    
    def generate_actionable_recommendations(self, risk_profile: Dict,
                                           current_status: Dict) -> List[str]:
        """
        Generate actionable recommendations
        
        Args:
            risk_profile: Employee risk profile
            current_status: Current status and metrics
        
        Returns:
            List of recommendations
        """
        prompt = f"""
Based on the following risk profile and current status, generate 5 specific, 
actionable recommendations for management.

Risk Profile:
- Delay Risk: {risk_profile.get('delay_risk', 0):.1%}
- Burnout Risk: {risk_profile.get('burnout_risk', 0):.1%}
- Cluster: {risk_profile.get('cluster', 'Unknown')}

Current Status:
- Working Hours: {current_status.get('working_hours', 0)} per week
- Overtime: {current_status.get('overtime_hours', 0)} hours
- Tasks Completed: {current_status.get('tasks_completed', 0)}
- Bug Count: {current_status.get('bug_count', 0)}

Generate specific, actionable recommendations that can be implemented immediately.
Format as a numbered list.
"""
        response = self._call_llm(prompt)
        
        if response:
            # Parse recommendations from response
            lines = response.split('\n')
            recommendations = [line.strip() for line in lines 
                             if line.strip() and line[0].isdigit()]
            return recommendations
        
        return []
    
    @staticmethod
    def _build_risk_report_prompt(employee_name: str,
                                 predictions: Dict,
                                 cluster_info: Dict,
                                 metrics: Dict) -> str:
        """Build prompt for individual risk report"""
        return f"""
Generate a professional risk analysis report for the following employee:

Employee: {employee_name}
Cluster Classification: {cluster_info.get('label', 'Unknown')}

Risk Assessment:
- Delay Risk Probability: {predictions.get('delay_risk', 0):.1%}
- Burnout Risk Probability: {predictions.get('burnout_risk', 0):.1%}
- Performance Score: {predictions.get('performance_score', 0):.1f}/100

Current Metrics:
- Working Hours/Week: {metrics.get('working_hours', 0)}
- Overtime Hours: {metrics.get('overtime_hours', 0)}
- Meeting Hours: {metrics.get('meeting_hours', 0)}
- Tasks Completed: {metrics.get('tasks_completed', 0)}
- Average Task Time: {metrics.get('avg_task_time', 0):.1f} hours
- Bug Count: {metrics.get('bug_count', 0)}
- Focus Score: {metrics.get('focus_score', 0):.1f}/100
- Deadline Gap: {metrics.get('deadline_gap', 0):.1f} days

Please provide:
1. Executive Summary
2. Risk Analysis (delay and burnout)
3. Performance Assessment
4. Key Findings
5. Recommended Actions
6. Success Metrics

Keep the report professional, data-driven, and actionable.
"""
    
    @staticmethod
    def _build_team_report_prompt(team_name: str,
                                 team_metrics: Dict,
                                 cluster_distribution: Dict,
                                 risk_summary: Dict) -> str:
        """Build prompt for team report"""
        return f"""
Generate a comprehensive team performance and risk report for {team_name}:

Team Size: {team_metrics.get('total_employees', 0)}
Average Performance Score: {team_metrics.get('avg_performance', 0):.1f}/100

Cluster Distribution:
{chr(10).join([f"- {cluster}: {count} employees" 
              for cluster, count in cluster_distribution.items()])}

Risk Summary:
- Employees at High Delay Risk: {risk_summary.get('high_delay_risk', 0)}
- Employees at High Burnout Risk: {risk_summary.get('high_burnout_risk', 0)}
- Average Workload: {team_metrics.get('avg_workload', 0):.1f}% capacity

Include:
1. Team Overview
2. Risk Assessment
3. Strengths and Achievements
4. Areas for Improvement
5. Resource Recommendations
6. Team Development Plan
"""
    
    @staticmethod
    def _build_organizational_report_prompt(organization_metrics: Dict,
                                           risk_trends: Dict,
                                           recommendations: List[str]) -> str:
        """Build prompt for organizational report"""
        return f"""
Generate an executive organizational report with the following data:

Organization Metrics:
- Total Employees: {organization_metrics.get('total_employees', 0)}
- Average Performance Score: {organization_metrics.get('avg_performance', 0):.1f}/100
- Overall Productivity: {organization_metrics.get('avg_productivity', 0):.1f}%

Risk Trends:
- Trend Direction: {risk_trends.get('trend', 'stable')}
- Critical Risk Count: {risk_trends.get('critical_count', 0)} employees
- High Risk Count: {risk_trends.get('high_count', 0)} employees

Strategic Recommendations:
{chr(10).join([f"- {rec}" for rec in recommendations[:5]])}

Generate:
1. Executive Summary
2. Organizational Health Assessment
3. Risk Analysis and Trends
4. Strategic Priorities
5. Resource Allocation Recommendations
6. Success Metrics and KPIs
7. Conclusion

Focus on strategic insights and actionable items for executives.
"""
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """
        Call LLM API
        
        Args:
            prompt: Prompt text
        
        Returns:
            Generated response or None
        """
        if not self.client:
            logger.warning("LLM client not initialized. Returning template response.")
            return self._generate_template_response(prompt)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert HR analytics consultant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return self._generate_template_response(prompt)
    
    @staticmethod
    def _generate_template_response(prompt: str) -> str:
        """
        Generate template response when LLM is not available
        Useful for development and testing
        """
        if "team performance" in prompt.lower():
            return """
TEAM PERFORMANCE ANALYSIS REPORT

Executive Summary:
The team is demonstrating strong overall performance with balanced workload distribution.
Current metrics indicate sustainable productivity levels with manageable risk factors.

Risk Assessment:
- Delay Risk: 2-3 employees showing elevated risk
- Burnout Risk: 1-2 employees require monitoring
- Workload Distribution: Relatively balanced across the team

Strengths:
- Consistent task completion rates
- Good focus scores indicating engagement
- Stable performance trends

Recommendations:
1. Monitor overtime patterns to prevent burnout
2. Implement workload balancing strategies
3. Increase check-ins for at-risk employees
4. Celebrate team achievements and milestones
"""
        else:
            return """
EMPLOYEE RISK ANALYSIS REPORT

Executive Summary:
This report provides a comprehensive analysis of current performance and risk indicators.

Key Findings:
- Current Risk Level: Moderate
- Performance Trajectory: Stable
- Workload Status: Within capacity

Recommendations:
1. Continue current work patterns
2. Maintain regular check-ins
3. Monitor key performance indicators
4. Plan professional development activities
"""


# Utility function
def generate_summary_report(data: Dict) -> str:
    """
    Generate quick summary report
    
    Args:
        data: Data dictionary with predictions and metrics
    
    Returns:
        Summary text
    """
    summary = f"""
PERFORMANCE SUMMARY
==================
Employee: {data.get('employee_name', 'Unknown')}
Date: {data.get('date', 'N/A')}

Risk Indicators:
- Delay Risk: {data.get('delay_risk', 0):.1%}
- Burnout Risk: {data.get('burnout_risk', 0):.1%}
- Performance Score: {data.get('performance_score', 0):.1f}/100

Current Workload:
- Working Hours: {data.get('working_hours', 0)} hrs/week
- Overtime: {data.get('overtime_hours', 0)} hrs
- Tasks Completed: {data.get('tasks_completed', 0)}

Trend: {data.get('trend', 'Stable')}
Status: {data.get('status', 'Normal')}
"""
    return summary
