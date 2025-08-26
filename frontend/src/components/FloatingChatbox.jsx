import { 
  MessageCircle, 
  X, 
  Minus, 
  Maximize2, 
  Minimize2, 
  Send, 
  Paperclip, 
  Image,
  File,
  Smile,
  Bot,
  User,
  Clock,
  Globe,
  BookOpen
} from 'lucide-react'
import { useState, useEffect, useRef, useCallback, useMemo } from 'react'

const API_BASE = 'http://localhost:8000'

const FloatingChatbox = () => {
  const [chatState, setChatState] = useState('mini') // mini, expanded, maximized
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isConnected, setIsConnected] = useState(true) // Set to true for demo
  const [selectedFiles, setSelectedFiles] = useState([])
  const [showFileMenu, setShowFileMenu] = useState(false)
  const [hasAddedWelcome, setHasAddedWelcome] = useState(false)
  
  const chatContainerRef = useRef(null)
  const fileInputRef = useRef(null)
  const imageInputRef = useRef(null)

  // Memoize API base to prevent unnecessary re-renders
  const apiBase = useMemo(() => API_BASE, [])

  // Check API health with useCallback to prevent recreation
  const checkHealth = useCallback(async () => {
    try {
      const response = await fetch(`${apiBase}/api/v1/health`)
      const data = await response.json()
      setIsConnected(data.status === 'healthy')
    } catch (error) {
      setIsConnected(false)
      console.error('Health check failed:', error)
    }
  }, [apiBase])

  useEffect(() => {
    checkHealth()
    const interval = setInterval(checkHealth, 60000)
    return () => clearInterval(interval)
  }, [checkHealth])

  // Fixed state change handler
  const handleStateChange = useCallback((newState) => {
    setChatState(newState)
  }, [])

  // Handle minimize - return to mini bubble
  const handleMinimize = useCallback(() => {
    setChatState('mini')
  }, [])

  // Handle close - reset chat and return to mini
  const handleClose = useCallback(() => {
    setMessages([])
    setHasAddedWelcome(false)
    setChatState('mini')
  }, [])

  // Handle maximize/restore toggle
  const handleMaximizeToggle = useCallback(() => {
    setChatState(chatState === 'maximized' ? 'expanded' : 'maximized')
  }, [chatState])

  // Handle welcome message when first expanded
  useEffect(() => {
    if (chatState === 'expanded' && messages.length === 0 && !hasAddedWelcome) {
      setMessages([{
        id: 1,
        content: "Hello! I'm CLAIRE, your BPI virtual assistant. How can I help you with your banking needs today?",
        isUser: false,
        timestamp: new Date()
      }])
      setHasAddedWelcome(true)
    }
  }, [chatState, messages.length, hasAddedWelcome])

  // Optimized auto scroll
  const scrollToBottom = useCallback(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
    }
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  // Memoized message handler to prevent recreations
  const handleSendMessage = useCallback(async () => {
    if ((!inputValue.trim() && selectedFiles.length === 0) || isLoading || !isConnected) return

    const userMessage = {
      id: Date.now(),
      content: inputValue.trim() || '📎 File attachment',
      isUser: true,
      timestamp: new Date(),
      files: selectedFiles
    }

    // Batch state updates for better performance
    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setSelectedFiles([])
    setIsLoading(true)

    // Add thinking message
    const thinkingMessage = {
      id: Date.now() + 1,
      content: '',
      isUser: false,
      isThinking: true,
      timestamp: new Date()
    }
    
    setMessages(prev => [...prev, thinkingMessage])

    try {
      const response = await fetch(`${apiBase}/api/v1/chat/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userMessage.content,
          session_id: 'bpi-user-123'
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      // Remove thinking message and add bot response
      setMessages(prev => {
        const filtered = prev.filter(msg => !msg.isThinking)
        return [...filtered, {
          id: Date.now() + 2,
          content: data.answer,
          isUser: false,
          timestamp: new Date(),
          metadata: {
            language: data.language,
            emotion: data.emotion,
            processing_time: data.processing_time,
            contexts: data.contexts
          }
        }]
      })

    } catch (error) {
      console.error('Error:', error)
      setMessages(prev => {
        const filtered = prev.filter(msg => !msg.isThinking)
        return [...filtered, {
          id: Date.now() + 2,
          content: `I apologize, but I'm having trouble connecting to our servers right now. Please try again in a moment.`,
          isUser: false,
          isError: true,
          timestamp: new Date()
        }]
      })
    } finally {
      setIsLoading(false)
    }
  }, [inputValue, selectedFiles, isLoading, isConnected, apiBase])

  // Optimized key press handler
  const handleKeyPress = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }, [handleSendMessage])

  // Optimized input change handler
  const handleInputChange = useCallback((e) => {
    const value = e.target.value
    setInputValue(value)
    
    // Auto-resize textarea
    const textarea = e.target
    textarea.style.height = 'auto'
    textarea.style.height = Math.min(textarea.scrollHeight, 100) + 'px'
  }, [])

  // Memoized file handlers
  const handleFileSelect = useCallback((type) => {
    if (type === 'image') {
      imageInputRef.current?.click()
    } else {
      fileInputRef.current?.click()
    }
    setShowFileMenu(false)
  }, [])

  const handleFileChange = useCallback((e, type) => {
    const files = Array.from(e.target.files)
    const newFiles = files.map(file => ({
      id: Date.now() + Math.random(),
      file,
      type,
      name: file.name,
      size: file.size
    }))
    
    setSelectedFiles(prev => [...prev, ...newFiles])
  }, [])

  const removeFile = useCallback((fileId) => {
    setSelectedFiles(prev => prev.filter(f => f.id !== fileId))
  }, [])

  // Memoized utility functions
  const getEmotionEmoji = useCallback((emotion) => {
    const emojis = {
      'confused': '😕',
      'frustrated': '😤', 
      'grateful': '😊',
      'neutral': '😐',
      'urgent': '⚠️',
      'worried': '😟'
    }
    return emojis[emotion] || '😐'
  }, [])

  const formatFileSize = useCallback((bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }, [])

  // Memoized components to prevent unnecessary re-renders
  const MessageList = useMemo(() => {
    return messages.map((message) => (
      <div key={message.id} className={`message ${message.isUser ? 'user' : 'bot'}`}>
        <div className="message-avatar">
          {message.isUser ? (
            <User className="w-4 h-4" />
          ) : (
            <Bot className="w-4 h-4" />
          )}
        </div>
        
        <div className="message-content">
          <div className={`message-bubble ${message.isError ? 'error' : ''}`} style={{
            background: message.isUser 
              ? 'linear-gradient(135deg, rgba(59, 130, 246, 0.9), rgba(147, 51, 234, 0.9))' 
              : message.isError 
                ? 'rgba(254, 242, 242, 0.9)' 
                : 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(20px)',
            border: message.isUser 
              ? '1px solid rgba(255, 255, 255, 0.3)' 
              : message.isError 
                ? '1px solid rgba(239, 68, 68, 0.3)' 
                : '1px solid rgba(229, 231, 235, 0.8)',
            borderRadius: '18px',
            padding: '0.875rem 1.125rem',
            boxShadow: message.isUser 
              ? '0 8px 25px rgba(59, 130, 246, 0.25)' 
              : '0 4px 20px rgba(0, 0, 0, 0.08)',
            color: message.isUser ? 'white' : message.isError ? '#dc2626' : '#1f2937',
            textAlign: 'left'
          }}>
            {message.isThinking ? (
              <div className="thinking-animation">
                <span>Thinking</span>
                <div className="dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            ) : (
              <>
                {message.content}
                {message.files && message.files.length > 0 && (
                  <div className="message-files">
                    {message.files.map(file => (
                      <div key={file.id} className="file-attachment">
                        {file.type === 'image' ? <Image className="w-4 h-4" /> : <File className="w-4 h-4" />}
                        <span>{file.name}</span>
                        <small>({formatFileSize(file.size)})</small>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>

          {/* Metadata */}
          {message.metadata && (
            <div className="message-metadata">
              {message.metadata.emotion && (
                <span className="metadata-tag" style={{
                  background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(236, 72, 153, 0.1))',
                  color: '#7c3aed',
                  borderRadius: '12px'
                }}>
                  {getEmotionEmoji(message.metadata.emotion.emotion)}
                  {message.metadata.emotion.emotion}
                </span>
              )}
              
              {message.metadata.language && (
                <span className="metadata-tag" style={{
                  background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(147, 51, 234, 0.1))',
                  color: '#3b82f6',
                  borderRadius: '12px'
                }}>
                  <Globe className="w-3 h-3" />
                  {message.metadata.language.language}
                </span>
              )}

              {message.metadata.processing_time && (
                <span className="metadata-tag" style={{
                  background: 'rgba(156, 163, 175, 0.15)',
                  color: '#6b7280',
                  borderRadius: '12px'
                }}>
                  <Clock className="w-3 h-3" />
                  {message.metadata.processing_time.toFixed(1)}s
                </span>
              )}

              {message.metadata.contexts && message.metadata.contexts.length > 0 && (
                <div className="sources">
                  <BookOpen className="w-3 h-3" />
                  <span>{message.metadata.contexts.length} source(s)</span>
                </div>
              )}
            </div>
          )}

          <div className="message-time">
            {message.timestamp.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
          </div>
        </div>
      </div>
    ))
  }, [messages, formatFileSize, getEmotionEmoji])

  // Memoized file preview
  const FilePreview = useMemo(() => {
    if (selectedFiles.length === 0) return null
    
    return (
      <div className="file-preview" style={{
        background: 'linear-gradient(135deg, rgba(249, 250, 251, 0.9), rgba(243, 244, 246, 0.8))',
        backdropFilter: 'blur(20px)',
        borderTop: '1px solid rgba(229, 231, 235, 0.6)'
      }}>
        {selectedFiles.map(file => (
          <div key={file.id} className="file-item" style={{
            background: 'rgba(255, 255, 255, 0.8)',
            borderRadius: '12px',
            border: '1px solid rgba(229, 231, 235, 0.6)',
            backdropFilter: 'blur(10px)'
          }}>
            {file.type === 'image' ? <Image className="w-4 h-4" /> : <File className="w-4 h-4" />}
            <span>{file.name}</span>
            <button onClick={() => removeFile(file.id)} className="remove-file">
              <X className="w-3 h-3" />
            </button>
          </div>
        ))}
      </div>
    )
  }, [selectedFiles, removeFile])

  // Render mini state (bubble)
  if (chatState === 'mini') {
    return (
      <div style={{
        position: 'fixed',
        bottom: '30px',
        right: '30px',
        zIndex: 10000
      }}>
        <button 
          onClick={() => handleStateChange('expanded')}
          title="Chat with CLAIRE"
          style={{
            width: '64px',
            height: '64px',
            borderRadius: '50%',
            background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.95), rgba(147, 51, 234, 0.9))',
            backdropFilter: 'blur(20px)',
            border: '1px solid rgba(255, 255, 255, 0.3)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            boxShadow: '0 10px 40px rgba(59, 130, 246, 0.4), 0 4px 15px rgba(0, 0, 0, 0.1)',
            color: 'white',
            position: 'relative'
          }}
        >
          <MessageCircle className="w-7 h-7" />
          <div style={{
            position: 'absolute',
            top: '6px',
            right: '6px',
            width: '14px',
            height: '14px',
            background: '#ef4444',
            borderRadius: '50%',
            border: '2px solid white',
            animation: 'pulse 2s infinite'
          }}></div>
        </button>
      </div>
    )
  }

  // Render expanded or maximized state
  return (
    <div style={{
      position: 'fixed',
      bottom: chatState === 'maximized' ? '0' : '30px',
      right: chatState === 'maximized' ? '0' : '30px',
      top: chatState === 'maximized' ? '0' : 'auto',
      left: chatState === 'maximized' ? '0' : 'auto',
      zIndex: 10000,
      background: 'rgba(255, 255, 255, 0.85)',
      backdropFilter: 'blur(40px)',
      border: chatState === 'maximized' ? 'none' : '1px solid rgba(229, 231, 235, 0.6)',
      borderRadius: chatState === 'maximized' ? '0' : '28px',
      boxShadow: chatState === 'maximized' ? 'none' : '0 25px 70px rgba(0, 0, 0, 0.15), 0 10px 30px rgba(0, 0, 0, 0.08)',
      width: chatState === 'maximized' ? '100vw' : '420px',
      height: chatState === 'maximized' ? '100vh' : '620px',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden'
    }}>
      {/* Header */}
      <div style={{
        padding: '1.75rem',
        background: 'linear-gradient(135deg, rgba(249, 250, 251, 0.9), rgba(243, 244, 246, 0.7))',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(229, 231, 235, 0.6)',
        display: 'flex',
        alignItems: 'center',
        gap: '1rem',
        textAlign: 'left'
      }}>
        <div style={{
          position: 'relative',
          width: '44px',
          height: '44px',
          background: 'linear-gradient(135deg, #10b981, #059669)',
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'white',
          boxShadow: '0 8px 25px rgba(16, 185, 129, 0.3)'
        }}>
          <Bot className="w-6 h-6" />
          <div style={{
            position: 'absolute',
            bottom: '1px',
            right: '1px',
            width: '14px',
            height: '14px',
            borderRadius: '50%',
            border: '2px solid white',
            background: isConnected ? '#10b981' : '#ef4444',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)'
          }}></div>
        </div>
        
        <div style={{ flex: 1, textAlign: 'left' }}>
          <h3 style={{
            fontSize: '1.2rem',
            fontWeight: 600,
            color: '#111827',
            margin: 0,
            textAlign: 'left'
          }}>CLAIRE</h3>
          <p style={{
            fontSize: '0.9rem',
            color: '#6b7280',
            margin: 0,
            textAlign: 'left'
          }}>
            {isConnected ? 'Online • BPI Assistant' : 'Reconnecting...'}
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button 
            onClick={handleMinimize}
            title="Minimize to bubble"
            style={{
              width: '36px',
              height: '36px',
              borderRadius: '12px',
              border: 'none',
              background: 'rgba(255, 255, 255, 0.8)',
              backdropFilter: 'blur(10px)',
              color: '#6b7280',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)'
            }}
          >
            <Minus className="w-4 h-4" />
          </button>
          <button 
            onClick={handleMaximizeToggle}
            title={chatState === 'maximized' ? 'Restore' : 'Maximize'}
            style={{
              width: '36px',
              height: '36px',
              borderRadius: '12px',
              border: 'none',
              background: 'rgba(255, 255, 255, 0.8)',
              backdropFilter: 'blur(10px)',
              color: '#6b7280',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)'
            }}
          >
            {chatState === 'maximized' ? 
              <Minimize2 className="w-4 h-4" /> : 
              <Maximize2 className="w-4 h-4" />
            }
          </button>
          <button 
            onClick={handleClose}
            title="Close and reset chat"
            style={{
              width: '36px',
              height: '36px',
              borderRadius: '12px',
              border: 'none',
              background: 'rgba(255, 255, 255, 0.8)',
              backdropFilter: 'blur(10px)',
              color: '#6b7280',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)'
            }}
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div 
        ref={chatContainerRef}
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '1.25rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.25rem',
          WebkitOverflowScrolling: 'touch',
          scrollbarWidth: 'thin',
          scrollbarColor: 'rgba(156, 163, 175, 0.4) transparent',
          textAlign: 'left'
        }}
      >
        {MessageList}
      </div>

      {/* File Attachments Preview */}
      {FilePreview}

      {/* Input Area */}
      <div style={{
        padding: '1.25rem',
        background: 'rgba(249, 250, 251, 0.9)',
        backdropFilter: 'blur(30px)',
        borderTop: '1px solid rgba(229, 231, 235, 0.6)',
        textAlign: 'left'
      }}>
        {!isConnected && (
          <div style={{
            background: 'rgba(251, 191, 36, 0.15)',
            border: '1px solid rgba(251, 191, 36, 0.3)',
            color: '#d97706',
            padding: '0.75rem 1rem',
            borderRadius: '12px',
            fontSize: '0.9rem',
            marginBottom: '1rem',
            textAlign: 'center',
            backdropFilter: 'blur(10px)'
          }}>
            Reconnecting to BPI servers...
          </div>
        )}

        <div style={{
          display: 'flex',
          alignItems: 'flex-end',
          gap: '0.75rem',
          position: 'relative'
        }}>
          <div style={{ position: 'relative' }}>
            <button 
              onClick={() => setShowFileMenu(!showFileMenu)}
              disabled={!isConnected}
              style={{
                width: '44px',
                height: '44px',
                borderRadius: '22px',
                border: 'none',
                background: 'rgba(243, 244, 246, 0.9)',
                backdropFilter: 'blur(10px)',
                color: '#6b7280',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)'
              }}
            >
              <Paperclip className="w-5 h-5" />
            </button>

            {showFileMenu && (
              <div style={{
                position: 'absolute',
                bottom: '52px',
                left: '0',
                background: 'rgba(255, 255, 255, 0.95)',
                backdropFilter: 'blur(30px)',
                border: '1px solid rgba(229, 231, 235, 0.6)',
                borderRadius: '16px',
                boxShadow: '0 15px 35px rgba(0, 0, 0, 0.15)',
                overflow: 'hidden',
                zIndex: 1000
              }}>
                <button 
                  onClick={() => handleFileSelect('image')}
                  style={{
                    background: 'none',
                    border: 'none',
                    padding: '0.875rem 1.25rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    cursor: 'pointer',
                    color: '#374151',
                    fontSize: '0.9rem',
                    width: '100%',
                    minWidth: '140px',
                    textAlign: 'left'
                  }}
                >
                  <Image className="w-4 h-4" />
                  Image
                </button>
                <button 
                  onClick={() => handleFileSelect('file')}
                  style={{
                    background: 'none',
                    border: 'none',
                    padding: '0.875rem 1.25rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    cursor: 'pointer',
                    color: '#374151',
                    fontSize: '0.9rem',
                    width: '100%',
                    minWidth: '140px',
                    textAlign: 'left'
                  }}
                >
                  <File className="w-4 h-4" />
                  File
                </button>
              </div>
            )}
          </div>

          <textarea
            value={inputValue}
            onChange={handleInputChange}
            onKeyPress={handleKeyPress}
            placeholder="Ask me anything about BPI services..."
            rows="1"
            disabled={!isConnected}
            style={{
              flex: 1,
              background: 'rgba(243, 244, 246, 0.9)',
              backdropFilter: 'blur(10px)',
              border: '2px solid rgba(229, 231, 235, 0.8)',
              borderRadius: '22px',
              padding: '0.875rem 1.25rem',
              fontSize: '0.95rem',
              fontFamily: 'inherit',
              resize: 'none',
              outline: 'none',
              minHeight: '44px',
              maxHeight: '100px',
              textAlign: 'left',
              color: '#111827',
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)'
            }}
          />

          <button
            onClick={handleSendMessage}
            disabled={(!inputValue.trim() && selectedFiles.length === 0) || isLoading || !isConnected}
            style={{
              width: '44px',
              height: '44px',
              borderRadius: '22px',
              border: 'none',
              background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
              color: 'white',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              opacity: ((!inputValue.trim() && selectedFiles.length === 0) || isLoading || !isConnected) ? 0.5 : 1,
              boxShadow: '0 8px 25px rgba(59, 130, 246, 0.3)'
            }}
          >
            {isLoading ? (
              <div style={{
                width: '20px',
                height: '20px',
                border: '2px solid rgba(255, 255, 255, 0.3)',
                borderTop: '2px solid white',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite'
              }} />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>

        {/* Hidden file inputs */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.doc,.docx,.txt,.xls,.xlsx"
          onChange={(e) => handleFileChange(e, 'file')}
          style={{ display: 'none' }}
          multiple
        />
        <input
          ref={imageInputRef}
          type="file"
          accept="image/*"
          onChange={(e) => handleFileChange(e, 'image')}
          style={{ display: 'none' }}
          multiple
        />
      </div>

      {/* Add CSS keyframes for animations that are still needed */}
      <style jsx>{`
        @keyframes pulse {
          0% { transform: scale(1); }
          50% { transform: scale(1.2); }
          100% { transform: scale(1); }
        }
        
        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        @keyframes bounce {
          0%, 80%, 100% { transform: scale(0); }
          40% { transform: scale(1); }
        }
      `}</style>
    </div>
  )
}

export default FloatingChatbox