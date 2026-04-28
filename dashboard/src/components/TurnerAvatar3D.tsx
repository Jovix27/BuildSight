import React from 'react'
import TurnerOrb3D from './TurnerOrb3D'
import './TurnerAvatar3D.css'

interface TurnerAvatar3DProps {
  isSpeaking?: boolean
  isThinking?: boolean
  size?: 'sm' | 'md' | 'lg'
}

const SIZE_PX = { sm: 80, md: 160, lg: 400 } as const

const TurnerAvatar3D: React.FC<TurnerAvatar3DProps> = ({
  isSpeaking = false,
  isThinking = false,
  size = 'md',
}) => {
  const state = isSpeaking ? 'speaking' : isThinking ? 'thinking' : 'idle'
  const px = SIZE_PX[size]

  return (
    <div
      className={`turner-avatar-3d turner-avatar-3d--${size} ${isSpeaking ? 'turner-avatar-3d--speaking' : ''} ${isThinking ? 'turner-avatar-3d--thinking' : ''}`}
      style={{ width: px, height: px }}
    >
      <TurnerOrb3D size={px} amplitude={isSpeaking ? 0.45 : 0} state={state} />
      <div className="turner-avatar-3d__overlay" />
    </div>
  )
}

export default TurnerAvatar3D
