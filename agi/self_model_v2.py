"""
Self-Model v2 (SM2) — AGI v8 Self-Awareness & Performance Tracking
Extended self-model that tracks missions, successes, failures, and performance evolution.
- Mission log (tasks attempted)
- Performance tracking (success rate, response time)
- Learning curve analysis
- Meta-cognition (knows its own limitations)
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
import json

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task completion status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


@dataclass
class Mission:
    """A task/mission the agent attempted."""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    successes: int = 0
    start_time: float = 0.0
    end_time: Optional[float] = None
    result: str = ""
    learning_gain: float = 0.0  # How much the agent improved on similar tasks
    
    @property
    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts
    
    @property
    def duration(self) -> float:
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


@dataclass
class PerformanceRecord:
    """Performance on a specific task/domain."""
    domain: str  # e.g., "reasoning", "memory_recall", "planning"
    accuracy: float
    response_time: float
    confidence: float
    timestamp: float = field(default_factory=time.time)


class SelfModel:
    """
    Comprehensive self-model tracking:
    - What the agent can do (capabilities)
    - Performance metrics (accuracy, speed, confidence)
    - Learning curve (improvement over time)
    - Meta-knowledge (knows its own strengths and weaknesses)
    """
    
    def __init__(self, agent_name: str = "AGI-v8"):
        self.agent_name = agent_name
        self.version = "v8"
        
        # Core capabilities
        self.capabilities = {
            "reasoning": 0.7,
            "memory_recall": 0.8,
            "planning": 0.6,
            "adaptation": 0.5,
            "self_awareness": 0.7,
        }
        
        # Mission tracking
        self.missions: Dict[str, Mission] = {}
        self.mission_count = 0
        
        # Performance tracking
        self.performance_records: Dict[str, List[PerformanceRecord]] = {
            domain: [] for domain in self.capabilities.keys()
        }
        
        # Cumulative stats
        self.total_tasks_attempted = 0
        self.total_tasks_succeeded = 0
        self.total_learning_gain = 0.0
        
        # Self-assessment
        self.known_limitations: List[str] = [
            "Limited world knowledge (trained on small dataset)",
            "Cannot physically interact",
            "Planning depth limited by compute",
            "Reasoning limited to implemented modules",
        ]
        
        self.known_strengths: List[str] = [
            "Fast memory recall",
            "Structured reasoning",
            "Continuous learning",
            "Self-aware limitations",
        ]
        
        self.confidence_calibration: Dict[str, float] = {}  # confidence -> actual accuracy
    
    def start_mission(self, description: str) -> str:
        """Create and start a new mission."""
        mission_id = f"mission_{self.mission_count}_{int(time.time() * 1000)}"
        self.mission_count += 1
        
        mission = Mission(
            id=mission_id,
            description=description,
            status=TaskStatus.IN_PROGRESS,
            start_time=time.time()
        )
        
        self.missions[mission_id] = mission
        self.total_tasks_attempted += 1
        
        logger.info("Started mission: %s | %s", mission_id, description)
        
        return mission_id
    
    def complete_mission(self, mission_id: str, status: TaskStatus, result: str = "", learning_gain: float = 0.0):
        """Mark a mission as complete."""
        if mission_id not in self.missions:
            logger.warning("Unknown mission: %s", mission_id)
            return
        
        mission = self.missions[mission_id]
        mission.status = status
        mission.end_time = time.time()
        mission.result = result
        mission.learning_gain = learning_gain
        
        if status == TaskStatus.SUCCESS:
            mission.successes += 1
            self.total_tasks_succeeded += 1
        
        mission.attempts += 1
        self.total_learning_gain += learning_gain
        
        logger.info("Completed mission: %s | Status: %s | Learning gain: %.2f", mission_id, status.value, learning_gain)
    
    def record_performance(self, domain: str, accuracy: float, response_time: float, confidence: float):
        """Record performance on a specific task."""
        if domain not in self.performance_records:
            self.performance_records[domain] = []
        
        record = PerformanceRecord(
            domain=domain,
            accuracy=accuracy,
            response_time=response_time,
            confidence=confidence
        )
        
        self.performance_records[domain].append(record)
        
        # Update confidence calibration
        conf_bucket = round(confidence * 4) / 4  # Bucket into 0.0, 0.25, 0.5, 0.75, 1.0
        if conf_bucket not in self.confidence_calibration:
            self.confidence_calibration[conf_bucket] = []
        self.confidence_calibration[conf_bucket].append(accuracy)
        
        # Update capability score (exponential moving average)
        current_capability = self.capabilities.get(domain, 0.5)
        self.capabilities[domain] = 0.9 * current_capability + 0.1 * accuracy
    
    def update_capabilities(self, domain: str, improvement: float):
        """Update a capability based on learning."""
        if domain not in self.capabilities:
            self.capabilities[domain] = 0.5
        
        self.capabilities[domain] = min(1.0, self.capabilities[domain] + improvement * 0.1)
    
    def get_learning_curve(self, domain: str, window: int = 10) -> List[float]:
        """Get performance trend over last N records."""
        if domain not in self.performance_records:
            return []
        
        records = self.performance_records[domain][-window:]
        return [r.accuracy for r in records]
    
    def estimate_task_difficulty(self, task_description: str) -> float:
        """Estimate difficulty of a task (0.0 = easy, 1.0 = hard) based on keywords."""
        difficulty = 0.5  # default
        
        hard_keywords = ["complex", "multi-step", "uncertain", "causal", "abstract", "optimize"]
        easy_keywords = ["factual", "retrieval", "simple", "template", "lookup"]
        
        for keyword in hard_keywords:
            if keyword in task_description.lower():
                difficulty = min(1.0, difficulty + 0.2)
        
        for keyword in easy_keywords:
            if keyword in task_description.lower():
                difficulty = max(0.0, difficulty - 0.2)
        
        return difficulty
    
    def estimate_success_probability(self, task_description: str) -> float:
        """Estimate probability of success on a task."""
        difficulty = self.estimate_task_difficulty(task_description)
        
        # Find relevant capability
        relevant_capability = 0.6
        if "reason" in task_description.lower():
            relevant_capability = self.capabilities.get("reasoning", 0.6)
        elif "memory" in task_description.lower() or "recall" in task_description.lower():
            relevant_capability = self.capabilities.get("memory_recall", 0.6)
        elif "plan" in task_description.lower():
            relevant_capability = self.capabilities.get("planning", 0.6)
        
        # P(success) = capability * (1 - difficulty)
        probability = relevant_capability * (1.0 - difficulty * 0.5)
        
        return min(1.0, max(0.0, probability))
    
    def get_self_assessment(self) -> Dict[str, Any]:
        """Generate comprehensive self-assessment."""
        success_rate = self.total_tasks_succeeded / max(1, self.total_tasks_attempted)
        
        return {
            "agent_name": self.agent_name,
            "version": self.version,
            "overall_success_rate": float(success_rate),
            "total_tasks_attempted": self.total_tasks_attempted,
            "total_tasks_succeeded": self.total_tasks_succeeded,
            "total_learning_gain": float(self.total_learning_gain),
            "capabilities": {k: float(v) for k, v in self.capabilities.items()},
            "known_strengths": self.known_strengths,
            "known_limitations": self.known_limitations,
            "performance_domains": {
                domain: {
                    "avg_accuracy": float(np.mean([r.accuracy for r in records])) if records else 0.0,
                    "avg_confidence": float(np.mean([r.confidence for r in records])) if records else 0.0,
                    "avg_response_time": float(np.mean([r.response_time for r in records])) if records else 0.0,
                    "num_records": len(records)
                }
                for domain, records in self.performance_records.items()
            }
        }
    
    def get_recent_missions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get summary of recent missions."""
        recent = sorted(
            self.missions.values(),
            key=lambda m: m.start_time,
            reverse=True
        )[:limit]
        
        return [
            {
                "id": m.id,
                "description": m.description,
                "status": m.status.value,
                "success_rate": float(m.success_rate),
                "duration": float(m.duration),
                "learning_gain": float(m.learning_gain),
            }
            for m in recent
        ]
    
    def identify_strengths_and_weaknesses(self) -> Dict[str, List[str]]:
        """Identify which domains are strengths vs weaknesses."""
        strengths = []
        weaknesses = []
        
        for domain, score in self.capabilities.items():
            if score > 0.7:
                strengths.append(f"{domain} ({score:.2f})")
            elif score < 0.5:
                weaknesses.append(f"{domain} ({score:.2f})")
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "areas_for_improvement": weaknesses,
        }
    
    def __repr__(self) -> str:
        return f"SelfModel({self.agent_name} | Success: {self.total_tasks_succeeded}/{self.total_tasks_attempted} | Capabilities: {len(self.capabilities)})"


# Import numpy for calculations
import numpy as np
