import React from 'react';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

/**
 * Clean markdown formatter for academic responses and chatbot outputs.
 * Parses headers, bold text, italics, bullet points, numbered lists, blockquotes,
 * code blocks, and inline code without external npm dependencies.
 */
export default function MarkdownRenderer({ content, className = '' }: MarkdownRendererProps) {
  if (!content) return null;

  // Split content into blocks by double newlines or single newlines
  const lines = content.split('\n');
  const elements: React.ReactNode[] = [];

  let inList = false;
  let listItems: React.ReactNode[] = [];
  let inCodeBlock = false;
  let codeBlockLines: string[] = [];

  const flushList = () => {
    if (inList && listItems.length > 0) {
      elements.push(
        <ul key={`ul-${elements.length}`} className="list-disc list-inside space-y-1 my-2 text-inherit">
          {listItems}
        </ul>
      );
      listItems = [];
      inList = false;
    }
  };

  const flushCodeBlock = () => {
    if (inCodeBlock && codeBlockLines.length > 0) {
      elements.push(
        <pre key={`code-${elements.length}`} className="p-3 bg-slate-900 text-slate-100 rounded-lg text-xs font-mono overflow-x-auto my-2.5">
          <code>{codeBlockLines.join('\n')}</code>
        </pre>
      );
      codeBlockLines = [];
      inCodeBlock = false;
    }
  };

  const parseInline = (text: string): React.ReactNode => {
    // Split by inline code, bold, italic
    const parts: React.ReactNode[] = [];
    let remaining = text;
    let idx = 0;

    // Regex for bold (**text** or __text__), code (`code`), italic (*text*)
    const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
    let lastIndex = 0;
    let match;

    while ((match = pattern.exec(text)) !== null) {
      // Add preceding plain text
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }

      const token = match[0];
      if (token.startsWith('**') && token.endsWith('**')) {
        parts.push(<strong key={idx++} className="font-semibold text-inherit">{token.slice(2, -2)}</strong>);
      } else if (token.startsWith('`') && token.endsWith('`')) {
        parts.push(
          <code key={idx++} className="px-1.5 py-0.5 bg-slate-200/80 text-indigo-700 font-mono text-xs rounded">
            {token.slice(1, -1)}
          </code>
        );
      } else if (token.startsWith('*') && token.endsWith('*')) {
        parts.push(<em key={idx++} className="italic text-inherit">{token.slice(1, -1)}</em>);
      }

      lastIndex = pattern.lastIndex;
    }

    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }

    return parts.length > 0 ? parts : text;
  };

  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i];
    const trimmed = rawLine.trim();

    // Code block toggle
    if (trimmed.startsWith('```')) {
      if (inCodeBlock) {
        flushCodeBlock();
      } else {
        flushList();
        inCodeBlock = true;
      }
      continue;
    }

    if (inCodeBlock) {
      codeBlockLines.push(rawLine);
      continue;
    }

    // Blank line
    if (!trimmed) {
      flushList();
      continue;
    }

    // Headers
    if (trimmed.startsWith('### ')) {
      flushList();
      elements.push(
        <h3 key={`h3-${i}`} className="text-base font-bold text-gray-900 mt-4 mb-1.5">
          {parseInline(trimmed.slice(4))}
        </h3>
      );
    } else if (trimmed.startsWith('## ')) {
      flushList();
      elements.push(
        <h2 key={`h2-${i}`} className="text-lg font-bold text-gray-900 mt-5 mb-2 border-b border-gray-200/60 pb-1">
          {parseInline(trimmed.slice(3))}
        </h2>
      );
    } else if (trimmed.startsWith('# ')) {
      flushList();
      elements.push(
        <h1 key={`h1-${i}`} className="text-xl font-extrabold text-gray-900 mt-6 mb-2">
          {parseInline(trimmed.slice(2))}
        </h1>
      );
    } else if (trimmed.startsWith('> ')) {
      // Blockquote
      flushList();
      elements.push(
        <blockquote key={`quote-${i}`} className="border-l-4 border-indigo-400 pl-3.5 py-1.5 my-2.5 bg-indigo-50/50 rounded-r text-gray-700 text-xs sm:text-sm italic">
          {parseInline(trimmed.slice(2))}
        </blockquote>
      );
    } else if (trimmed.startsWith('- ') || trimmed.startsWith('* ') || trimmed.startsWith('• ')) {
      // Bullet list item
      inList = true;
      listItems.push(
        <li key={`li-${i}`} className="text-xs sm:text-sm text-gray-800 leading-relaxed">
          {parseInline(trimmed.replace(/^[-*•]\s+/, ''))}
        </li>
      );
    } else if (/^\d+\.\s+/.test(trimmed)) {
      // Numbered list item
      flushList();
      const numMatch = trimmed.match(/^(\d+\.)\s+(.*)/);
      if (numMatch) {
        elements.push(
          <div key={`num-${i}`} className="flex gap-2 items-start my-1 text-xs sm:text-sm text-gray-800 leading-relaxed">
            <span className="font-semibold text-indigo-600 shrink-0">{numMatch[1]}</span>
            <span>{parseInline(numMatch[2])}</span>
          </div>
        );
      }
    } else {
      flushList();
      elements.push(
        <p key={`p-${i}`} className="text-xs sm:text-sm text-gray-800 leading-relaxed my-1.5">
          {parseInline(trimmed)}
        </p>
      );
    }
  }

  flushList();
  flushCodeBlock();

  return <div className={`space-y-1 ${className}`}>{elements}</div>;
}
