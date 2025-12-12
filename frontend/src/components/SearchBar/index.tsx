import React, { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { useHistory } from '@docusaurus/router';
import { searchContent, SearchResult } from '../../lib/search-utils';
import styles from './styles.module.css';

/**
 * Search Bar Component
 *
 * An enhanced search interface with modern UI elements and improved UX.
 * Features:
 * - Keyboard shortcuts (Cmd/Ctrl + K)
 * - Smooth animations and transitions
 * - Full-text search via Lunr.js
 * - Recent search history
 * - Improved visual hierarchy
 */
export function SearchBar(): React.ReactElement {
    const [isOpen, setIsOpen] = useState(false);
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<SearchResult[]>([]);
    const [loading, setLoading] = useState(false);
    const [recentSearches, setRecentSearches] = useState<string[]>([]);
    const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
    const [searchAvailable, setSearchAvailable] = useState<boolean>(true); // Track if search functionality is available
    const inputRef = useRef<HTMLInputElement>(null);
    const modalRef = useRef<HTMLDivElement>(null);
    const history = useHistory();

    // Check if search is available on component mount
    useEffect(() => {
        const checkSearchAvailability = async () => {
            try {
                // Try a simple search to see if the index is available
                const testResults = await searchContent("test");
                setSearchAvailable(testResults.length >= 0); // If no error, search is available
            } catch (error) {
                console.error('Search initialization error:', error);
                setSearchAvailable(false);
            }
        };

        checkSearchAvailability();
    }, []);

    // Handle keyboard shortcut (Cmd/Ctrl + K)
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault();
                setIsOpen(true);
                setTimeout(() => inputRef.current?.focus(), 100);
            }
            if (e.key === 'Escape' && isOpen) {
                setIsOpen(false);
                setQuery('');
                setSelectedIndex(null);
            }

            // Handle arrow keys for navigation
            if (isOpen && results.length > 0) {
                if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    setSelectedIndex(prev => {
                        const nextIndex = prev === null ? 0 : (prev < results.length - 1 ? prev + 1 : prev);
                        return nextIndex;
                    });
                } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    setSelectedIndex(prev => {
                        const prevIndex = prev === null ? results.length - 1 : (prev > 0 ? prev - 1 : null);
                        return prevIndex;
                    });
                } else if (e.key === 'Enter' && selectedIndex !== null && results[selectedIndex]) {
                    handleResultClick(results[selectedIndex].url || '');
                }
            }
        };

        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, [isOpen, results, selectedIndex]);

    // Close on outside click (clicking the overlay, not the modal)
    const handleOverlayClick = (e: React.MouseEvent) => {
        // Only close if clicking directly on the overlay, not on the modal
        if (e.target === e.currentTarget) {
            setIsOpen(false);
            setQuery('');
            setSelectedIndex(null);
        }
    };

    // Perform search using the search utility
    useEffect(() => {
        if (!isOpen || !query.trim()) {
            setResults([]);
            setLoading(false);
            return;
        }

        // Debounce search
        const timeoutId = setTimeout(async () => {
            setLoading(true);
            setSelectedIndex(null); // Reset selection when new search starts

            try {
                const searchResults = await searchContent(query);
                setResults(searchResults);

                // Add to recent searches if not already present
                if (query.trim() && !recentSearches.includes(query.trim()) && searchResults.length > 0) {
                    const newRecentSearches = [query.trim(), ...recentSearches.slice(0, 4)];
                    setRecentSearches(newRecentSearches);
                }
            } catch (error) {
                console.error('Search error:', error);
                // Silently handle errors - search may not be available in dev mode
                setResults([]);
            } finally {
                setLoading(false);
            }
        }, 300);

        return () => clearTimeout(timeoutId);
    }, [query, isOpen, recentSearches]);

    const handleResultClick = (url: string) => {
        if (url) {
            setIsOpen(false);
            setQuery('');
            setSelectedIndex(null);
            history.push(url);
        }
    };

    const handleRecentSearchClick = (term: string) => {
        setQuery(term);
        inputRef.current?.focus();
    };

    const clearRecentSearches = () => {
        setRecentSearches([]);
    };

    return (
        <div className={styles.searchContainer}>
            {/* Search Trigger Button */}
            <button
                className={styles.searchTrigger}
                onClick={() => {
                    setIsOpen(!isOpen);
                    if (!isOpen) {
                        // Use a slightly longer delay to ensure the modal is rendered
                        setTimeout(() => {
                            inputRef.current?.focus();
                        }, 150);
                    }
                }}
                aria-label="Search docs"
                aria-expanded={isOpen}
            >
                <svg
                    className={styles.searchIcon}
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                >
                    <circle cx="11" cy="11" r="8" />
                    <path d="m21 21-4.35-4.35" />
                </svg>
                <span className={styles.searchLabel}>Search docs</span>
                <kbd className={styles.searchShortcut}>
                    <span className={styles.kbdKey}>⌘</span>
                    <span className={styles.kbdKey}>K</span>
                </kbd>
            </button>

            {/* Search Modal/Overlay - rendered via Portal to escape navbar stacking context */}
            {isOpen && typeof document !== 'undefined' && createPortal(
                <div className={styles.searchOverlay} onClick={handleOverlayClick}>
                    <div ref={modalRef} className={styles.searchModal}>
                        {/* Search Input */}
                        <div className={styles.searchInputWrapper}>
                            <svg
                                className={styles.searchInputIcon}
                                width="20"
                                height="20"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                            >
                                <circle cx="11" cy="11" r="8" />
                                <path d="m21 21-4.35-4.35" />
                            </svg>
                            <input
                                ref={inputRef}
                                type="text"
                                className={styles.searchInput}
                                placeholder="Search documentation, articles, and resources..."
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                autoFocus
                            />
                            {query && (
                                <button
                                    className={styles.clearButton}
                                    onClick={() => setQuery('')}
                                    aria-label="Clear search"
                                >
                                    <svg
                                        width="16"
                                        height="16"
                                        viewBox="0 0 24 24"
                                        fill="none"
                                        stroke="currentColor"
                                        strokeWidth="2"
                                    >
                                        <line x1="18" y1="6" x2="6" y2="18" />
                                        <line x1="6" y1="6" x2="18" y2="18" />
                                    </svg>
                                </button>
                            )}
                        </div>

                        {/* Search Results */}
                        <div className={styles.searchResults}>
                            {loading && query && (
                                <div className={styles.searchLoading}>
                                    <div className={styles.loadingSpinner} />
                                    <span>Searching through documentation...</span>
                                </div>
                            )}

                            {!loading && query && results.length === 0 && searchAvailable && (
                                <div className={styles.searchEmpty}>
                                    <svg
                                        width="48"
                                        height="48"
                                        viewBox="0 0 24 24"
                                        fill="none"
                                        stroke="currentColor"
                                        strokeWidth="1.5"
                                    >
                                        <circle cx="11" cy="11" r="8" />
                                        <path d="m21 21-4.35-4.35" />
                                    </svg>
                                    <p>No results found</p>
                                    <span className={styles.searchEmptyHint}>
                                        Try different keywords or check your spelling
                                    </span>
                                </div>
                            )}

                            {!loading && query && results.length === 0 && !searchAvailable && (
                                <div className={styles.searchEmpty}>
                                    <svg
                                        width="48"
                                        height="48"
                                        viewBox="0 0 24 24"
                                        fill="none"
                                        stroke="currentColor"
                                        strokeWidth="1.5"
                                    >
                                        <circle cx="11" cy="11" r="8" />
                                        <path d="m21 21-4.35-4.35" />
                                    </svg>
                                    <p>Search not available</p>
                                    <span className={styles.searchEmptyHint}>
                                        Search functionality is only available in production builds
                                    </span>
                                </div>
                            )}

                            {!loading && query && results.length > 0 && (
                                <>
                                    <div className={styles.searchResultsHeader}>
                                        <span className={styles.resultsCount}>
                                            {results.length}{' '}
                                            {results.length === 1 ? 'result' : 'results'} for "{query}"
                                        </span>
                                    </div>
                                    <div className={styles.resultsList}>
                                        {results.map((result, index: number) => (
                                            <button
                                                key={`${index}-${result.title || result.url || index}`}
                                                className={`${styles.resultItem} ${selectedIndex === index ? styles.resultItemSelected : ''}`}
                                                onClick={() =>
                                                    handleResultClick(result.url || '')
                                                }
                                                onMouseEnter={() => setSelectedIndex(index)}
                                            >
                                                <div className={styles.resultHeader}>
                                                    <span className={styles.resultTitle}>
                                                        {result.title || result.url?.split('/').pop() || 'Untitled'}
                                                    </span>
                                                    {result.type && (
                                                        <span className={styles.resultType}>
                                                            {result.type}
                                                        </span>
                                                    )}
                                                </div>
                                                {result.text && (
                                                    <p className={styles.resultSnippet}>{result.text}</p>
                                                )}
                                                {result.url && (
                                                    <div className={styles.resultPath}>
                                                        {result.url.replace(/^\//, '').replace(/\/$/, '')}
                                                    </div>
                                                )}
                                            </button>
                                        ))}
                                    </div>
                                </>
                            )}

                            {!query && (
                                <div className={styles.searchSuggestions}>
                                    {!searchAvailable && (
                                        <div className={styles.searchUnavailableMessage}>
                                            <div className={styles.unavailableIcon}>
                                                <svg
                                                    width="24"
                                                    height="24"
                                                    viewBox="0 0 24 24"
                                                    fill="none"
                                                    stroke="currentColor"
                                                    strokeWidth="2"
                                                    strokeLinecap="round"
                                                    strokeLinejoin="round"
                                                >
                                                    <circle cx="12" cy="12" r="10" />
                                                    <line x1="15" y1="9" x2="9" y2="15" />
                                                    <line x1="9" y1="9" x2="15" y2="15" />
                                                </svg>
                                            </div>
                                            <h3 className={styles.suggestionsTitle}>Search Unavailable</h3>
                                            <p className={styles.unavailableText}>
                                                Search functionality is only available in production builds.
                                                The search index is generated during the build process.
                                            </p>
                                        </div>
                                    )}
                                    {recentSearches.length > 0 && searchAvailable && (
                                        <div className={styles.recentSearchesSection}>
                                            <div className={styles.suggestionsHeader}>
                                                <h3 className={styles.suggestionsTitle}>Recent Searches</h3>
                                                <button
                                                    className={styles.clearHistoryBtn}
                                                    onClick={clearRecentSearches}
                                                    aria-label="Clear recent searches"
                                                >
                                                    Clear
                                                </button>
                                            </div>
                                            <div className={styles.recentSearchesList}>
                                                {recentSearches.map((term, index) => (
                                                    <button
                                                        key={`recent-${index}`}
                                                        className={styles.recentSearchItem}
                                                        onClick={() => handleRecentSearchClick(term)}
                                                    >
                                                        <svg
                                                            className={styles.historyIcon}
                                                            width="16"
                                                            height="16"
                                                            viewBox="0 0 24 24"
                                                            fill="none"
                                                            stroke="currentColor"
                                                            strokeWidth="2"
                                                            strokeLinecap="round"
                                                            strokeLinejoin="round"
                                                        >
                                                            <circle cx="12" cy="12" r="10" />
                                                            <polyline points="12 6 12 12 16 14" />
                                                        </svg>
                                                        <span>{term}</span>
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                    {searchAvailable && (
                                        <div className={styles.searchTips}>
                                            <div className={styles.tipsGroup}>
                                                <h3 className={styles.suggestionsTitle}>Search Tips</h3>
                                                <ul className={styles.tipsList}>
                                                    <li className={styles.tipItem}>
                                                        <div className={styles.tipBullet}>•</div>
                                                        <span>Use quotes for exact phrase matching</span>
                                                    </li>
                                                    <li className={styles.tipItem}>
                                                        <div className={styles.tipBullet}>•</div>
                                                        <span>Type keywords to find related content</span>
                                                    </li>
                                                    <li className={styles.tipItem}>
                                                        <div className={styles.tipBullet}>•</div>
                                                        <span>Use arrow keys to navigate results</span>
                                                    </li>
                                                </ul>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                </div>,
                document.body
            )}
        </div>
    );
}