import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Clock, 
  ShieldCheck, 
  Activity,
  User,
  Cpu
} from 'lucide-react';
import { io, Socket } from 'socket.io-client';
import './SymphonyPanel.css';

interface SymphonyTask {
  id: number;
  issue_number: number;
  title: string;
  state: 'pending' | 'dispatched' | 'running' | 'verified' | 'failed' | 'completed';
  priority: 'high' | 'medium' | 'low';
  branch_name: string;
  assigned_agent: string;
  turns_count: number;
  created_at: string;
}

interface SymphonyEvent {
  id: number;
  task_id: number;
  event_type: string;
  message: string;
  payload: string;
  created_at: string;
}

export const SymphonyPanel: React.FC = () => {
  const [tasks, setTasks] = useState<SymphonyTask[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState<number | null>(null);
  const [events, setEvents] = useState<SymphonyEvent[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const socketRef = useRef<Socket | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const selectedTask = tasks.find(t => t.id === selectedTaskId);

  // Initial Fetch
  useEffect(() => {
    fetchTasks();
    const interval = setInterval(fetchTasks, 10000); // Poll every 10s as fallback
    return () => clearInterval(interval);
  }, []);

  const fetchTasks = async () => {
    try {
      const res = await fetch('http://localhost:3001/api/symphony/tasks');
      const data = await res.json();
      if (data.status === 'ok') {
        setTasks(data.tasks);
      }
    } catch (err) {
      console.error('Failed to fetch Symphony tasks', err);
    }
  };

  const fetchEvents = async (id: number) => {
    try {
      const res = await fetch(`http://localhost:3001/api/symphony/tasks/${id}/events`);
      const data = await res.json();
      if (data.status === 'ok') {
        setEvents(data.events);
      }
    } catch (err) {
      console.error('Failed to fetch Symphony events', err);
    }
  };

  useEffect(() => {
    if (selectedTaskId) {
      fetchEvents(selectedTaskId);
    }
  }, [selectedTaskId]);

  // Socket.io for Real-time
  useEffect(() => {
    const socket = io('http://localhost:3001', {
      transports: ['websocket'],
      auth: { token: '' }
    });

    socket.on('connect', () => setIsConnected(true));
    socket.on('disconnect', () => setIsConnected(false));

    socket.on('symphony_event', (data: any) => {
      if (data.taskId === selectedTaskId) {
        setEvents(prev => [...prev, {
          id: Date.now(),
          task_id: data.taskId,
          event_type: data.type,
          message: data.message,
          payload: typeof data.payload === 'object' ? JSON.stringify(data.payload, null, 2) : data.payload,
          created_at: data.created_at || new Date().toISOString()
        }]);
      }
      if (['dispatched', 'running', 'completed', 'failed'].includes(data.type)) {
        fetchTasks();
      }
    });

    socketRef.current = socket;
    return () => {
      socket.disconnect();
    };
  }, [selectedTaskId]);

  // Auto-scroll log
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events]);

  const handleApprove = async (id: number) => {
    try {
      await fetch(`http://localhost:3001/api/symphony/tasks/${id}/approve`, { method: 'POST' });
      fetchTasks();
    } catch (err) {
      console.error('Approval failed', err);
    }
  };

  return (
    <div className="god-panel god-flex-grow symphony-panel-container">
      {/* Sidebar: Task List */}
      <div className="symphony-sidebar">
        <div className="god-panel-header">
          <div className="symphony-status-indicator">
            <div className={`symphony-dot ${isConnected ? 'symphony-dot--online' : 'symphony-dot--offline'}`} />
            <h3>ORCHESTRATION</h3>
          </div>
          <Activity size={16} className="god-accent-icon" />
        </div>
        
        <div className="symphony-task-list">
          {tasks.map((task) => (
            <motion.div
              key={task.id}
              whileHover={{ x: 4 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setSelectedTaskId(task.id)}
              className={`symphony-task-card ${selectedTaskId === task.id ? 'symphony-task-card--active' : ''}`}
            >
              <div className="symphony-task-header">
                <span className={`symphony-status-badge status--${task.state}`}>
                  {task.state.toUpperCase()}
                </span>
                <span className={`symphony-priority-label priority--${task.priority}`}>
                  P:{task.priority.toUpperCase()}
                </span>
              </div>
              <h4 className="symphony-task-title">#{task.issue_number} {task.title}</h4>
              <div className="symphony-task-footer">
                <div className="symphony-task-meta-item">
                  <User size={10} />
                  <span>{task.assigned_agent || 'UNASSIGNED'}</span>
                </div>
                <div className="symphony-task-meta-item">
                  <Clock size={10} />
                  <span>{new Date(task.created_at).toLocaleTimeString()}</span>
                </div>
              </div>
            </motion.div>
          ))}
          {tasks.length === 0 && (
            <div className="symphony-empty-state">
              <Activity className="symphony-empty-icon" />
              <p>No active tasks found</p>
            </div>
          )}
        </div>
      </div>

      {/* Main Content: Event Log */}
      <div className="symphony-main">
        {selectedTask ? (
          <>
            <div className="symphony-detail-header">
              <div className="symphony-detail-top">
                <div>
                  <h2 className="symphony-detail-title">{selectedTask.title}</h2>
                  <p className="symphony-detail-branch">Branch: {selectedTask.branch_name}</p>
                </div>
                {selectedTask.state === 'pending' && (
                  <button 
                    onClick={() => handleApprove(selectedTask.id)}
                    className="god-btn god-btn-primary symphony-approve-btn"
                  >
                    <ShieldCheck size={16} />
                    APPROVE EXECUTION
                  </button>
                )}
              </div>
              
              <div className="symphony-stats-grid">
                <div className="symphony-stat-card">
                  <span className="symphony-stat-label">Turns</span>
                  <span className="symphony-stat-value">{selectedTask.turns_count} / 10</span>
                </div>
                <div className="symphony-stat-card">
                  <span className="symphony-stat-label">Agent</span>
                  <span className="symphony-stat-value" style={{ color: 'var(--color-silver-dark)' }}>{selectedTask.assigned_agent || 'NONE'}</span>
                </div>
                <div className="symphony-stat-card">
                  <span className="symphony-stat-label">Created</span>
                  <span className="symphony-stat-value" style={{ fontSize: '11px' }}>{new Date(selectedTask.created_at).toLocaleString()}</span>
                </div>
                <div className="symphony-stat-card">
                  <span className="symphony-stat-label">Isolation</span>
                  <span className="symphony-stat-value" style={{ color: '#00ff80', fontSize: '13px' }}>ENFORCED</span>
                </div>
              </div>
            </div>

            <div 
              ref={scrollRef}
              className="symphony-event-log"
            >
              <AnimatePresence initial={false}>
                {events.map((event) => (
                  <motion.div
                    key={event.id}
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="symphony-event-item"
                  >
                    <span className="symphony-event-time">
                      [{new Date(event.created_at).toLocaleTimeString()}]
                    </span>
                    <div className="symphony-event-content">
                      <span className={`symphony-event-type ${
                        event.event_type.includes('fail') || event.event_type.includes('error') ? 'status--failed' :
                        event.event_type.includes('passed') || event.event_type.includes('success') ? 'status--verified' :
                        'status--dispatched'
                      }`}>
                        {event.event_type.toUpperCase()}
                      </span>
                      <span className="symphony-event-message">{event.message}</span>
                      {event.payload && event.payload !== '{}' && (
                        <div className="symphony-event-payload">
                          <pre>{event.payload}</pre>
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
              {events.length === 0 && (
                <div className="symphony-empty-state" style={{ opacity: 0.3 }}>
                  <p>Telemetry stream active...</p>
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="symphony-empty-state">
            <Cpu className="symphony-empty-icon" />
            <p style={{ opacity: 0.5, fontSize: '0.8rem', letterSpacing: '0.1em' }}>SELECT TASK FOR TELEMETRY</p>
          </div>
        )}
      </div>
    </div>
  );
};

