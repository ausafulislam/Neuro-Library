import React, { useState, useRef, useEffect, FormEvent } from "react";
import type { JSX } from "react";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import styles from "./NeuroBot.module.css";
import useBaseUrl from '@docusaurus/useBaseUrl';
import MarkdownRenderer from "../MarkdownRenderer";

type Message = {
    role: "user" | "bot";
    text: string;
};

export default function NeuroBot(): JSX.Element {
    const logo = useBaseUrl('/img/logo.png');
    const { siteConfig } = useDocusaurusContext();
    const { neuroBot_api_key } = siteConfig.customFields;

    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState<Message[]>([
        {
            role: "bot",
            text: "Hi, I am the Neuro Library assistant. Ask me anything related to books.",
        },
    ]);
    const [input, setInput] = useState<string>("");
    const [loading, setLoading] = useState<boolean>(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const toggleChat = () => {
        setIsOpen((prev) => !prev);
    };

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const sendMessage = async (e: FormEvent) => {
        e.preventDefault();
        if (!input.trim() || loading) return;

        const userMessage: Message = { role: "user", text: input };
        setMessages((prev) => [...prev, userMessage]);
        setInput("");
        setLoading(true);

        try {
            const apiUrl = neuroBot_api_key as string;

            // Send POST request to your FastAPI endpoint
            const response = await fetch(apiUrl, {
                method: "POST",
                headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                body: new URLSearchParams({
                    message: userMessage.text,
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            const botMessage: Message = {
                role: "bot",
                text: data.response
            };
            setMessages((prev) => [...prev, botMessage]);

        } catch (error) {
            console.error("Error sending message:", error);
            const errorMessage: Message = {
                role: "bot",
                text: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`
            };
            setMessages((prev) => [...prev, errorMessage]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <div className={styles.chatToggle} onClick={toggleChat}>
                <img src={logo} alt="Assistant" className={styles.chatIcon} />
                {!isOpen && (
                    <div className={styles.tooltip}>
                        Try Assistant
                        <div className={styles.tooltiparrow}></div>
                    </div>
                )}
            </div>

            {isOpen && (
                <div className={styles.chatPanel}>
                    <div className={styles.chatHeader}>
                        Neuro Library Assistant
                        <button className={styles.closeButton} onClick={toggleChat}>
                            ✕
                        </button>
                    </div>

                    <div className={styles.chatMessages}>
                        {messages.map((msg, idx) => (
                            <div
                                key={idx}
                                className={
                                    msg.role === "user"
                                        ? styles.userMessage
                                        : styles.botMessage
                                }
                            >
                                {msg.role === "user" ? (
                                    msg.text
                                ) : (
                                    <MarkdownRenderer children={msg.text.replace(/(\[.*?\])/g, "$1\n")} />
                                )}
                            </div>
                        ))}
                        <div ref={messagesEndRef} />
                    </div>

                    <form className={styles.chatForm} onSubmit={sendMessage}>
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ask from the docs..."
                            disabled={loading}
                            autoFocus
                        />
                        <button type="submit" disabled={loading || !input.trim()}>
                            {loading ? "..." : "Send"}
                        </button>
                    </form>
                    <p className={styles.footer}>AI-generated, for reference only</p>
                </div>
            )}
        </>
    );
}