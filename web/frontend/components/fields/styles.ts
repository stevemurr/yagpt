import type React from 'react';

export const labelStyle: React.CSSProperties = {
  fontSize: '9px',
  color: '#999',
};

export const inputStyle: React.CSSProperties = {
  flex: 1,
  padding: '4px 6px',
  fontSize: '10px',
  fontFamily: 'inherit',
  border: '1px solid #e0e0e0',
  borderRadius: '3px',
  outline: 'none',
  color: '#333',
};

export const selectStyle: React.CSSProperties = {
  flex: 1,
  padding: '4px 6px',
  fontSize: '10px',
  fontFamily: 'inherit',
  border: '1px solid #e0e0e0',
  borderRadius: '3px',
  background: '#fafafa',
  color: '#555',
  cursor: 'pointer',
  outline: 'none',
};

export const sliderStyle: React.CSSProperties = {
  flex: 1,
  accentColor: '#999',
  height: '2px',
};

export const valueStyle: React.CSSProperties = {
  fontSize: '10px',
  fontWeight: 600,
  color: '#333',
  textAlign: 'right' as const,
};

export const btnStyle: React.CSSProperties = {
  padding: '8px',
  fontSize: '10px',
  fontWeight: 600,
  fontFamily: 'inherit',
  letterSpacing: '1px',
  background: '#333',
  color: '#fff',
  border: 'none',
  borderRadius: '3px',
  cursor: 'pointer',
};

export const stopBtnStyle: React.CSSProperties = {
  padding: '5px',
  fontSize: '9px',
  fontFamily: 'inherit',
  background: 'transparent',
  border: '1px solid #e0e0e0',
  borderRadius: '3px',
  color: '#999',
  cursor: 'pointer',
};

export const toggleBtnStyle = (active: boolean): React.CSSProperties => ({
  padding: '2px 6px',
  fontSize: '8px',
  fontFamily: 'inherit',
  background: active ? '#333' : '#f5f5f5',
  color: active ? '#fff' : '#888',
  border: `1px solid ${active ? '#333' : '#e8e8e8'}`,
  borderRadius: '2px',
  cursor: 'pointer',
});
