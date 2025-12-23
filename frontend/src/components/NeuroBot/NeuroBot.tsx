import React, { useState, useRef, useEffect, FormEvent } from "react";
import type { JSX } from "react";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import styles from "./NeuroBot.module.css";
import useBaseUrl from '@docusaurus/useBaseUrl';
import MarkdownRenderer from "../MarkdownRenderer";

type Message = {
    role: "user" | "bot" | "loader";
    text: string;
};

export default function NeuroBot(): JSX.Element {
    const logo = useBaseUrl('/img/logo.png');
    const { siteConfig } = useDocusaurusContext();
    const { neuroBot_api_key } = siteConfig.customFields;

    // for sound effect 
    const sendSoundRef = useRef<HTMLAudioElement | null>(null);
    const receiveSoundRef = useRef<HTMLAudioElement | null>(null);
    useEffect(() => {
        sendSoundRef.current = new Audio("/sounds/send.wav");
        receiveSoundRef.current = new Audio("/sounds/received.wav");
    }, []);

    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState<Message[]>([
        {
            role: "bot",
            text: "Hi, I am Neuro Library assistant. Ask me anything related to books.",
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
        setMessages(prev => [...prev, userMessage]);

        sendSoundRef.current?.play();

        setInput("");
        setLoading(true);

        // show dotted loader
        setMessages(prev => [
            ...prev,
            { role: "loader", text: "" }
        ]);

        try {
            const response = await fetch(neuroBot_api_key as string, {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: new URLSearchParams({ message: userMessage.text }),
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();

            // remove loader, add bot response
            setMessages(prev =>
                prev.filter(m => m.role !== "loader").concat({
                    role: "bot",
                    text: data.response,
                })
            );

            receiveSoundRef.current?.play();

        } catch (err) {
            setMessages(prev =>
                prev.filter(m => m.role !== "loader").concat({
                    role: "bot",
                    text: "Sorry, something went wrong. Please try again.",
                })
            );
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
                        {messages.map((msg, idx) => {
                            // Loader message
                            if (msg.role === "loader") {
                                return (
                                    <div key={idx} className={styles.botMessage}>
                                        <div className={styles.loader}></div>
                                    </div>
                                );
                            }

                            return (
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
                                        <MarkdownRenderer
                                            children={msg.text
                                                .replace(/(\[.*?\])/g, "$1\n")
                                                .replace(/(\r?\n)+/g, "\n")}
                                        />
                                    )}
                                </div>
                            );
                        })}

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

                    <p className={styles.footer}>
                        AI-generated, for reference only
                    </p>
                </div>
            )}
        </>
    );

}