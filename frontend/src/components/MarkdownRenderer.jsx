// import React from "react";
// import Markdown from "react-markdown";
// import remarkGfm from "remark-gfm";
// import rehypeRaw from "rehype-raw";
// import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
// import { dracula } from "react-syntax-highlighter/dist/cjs/styles/prism";

// export default function MarkdownRenderer({ children }) {
//     return (
//         <Markdown
//             remarkPlugins={[remarkGfm]}
//             rehypePlugins={[rehypeRaw]}
//             components={{
//                 code({ node, inline, className, children: codeChildren, ...props }) {
//                     const match = /language-(\w+)/.exec(className || "");

//                     return !inline && match ? (
//                         <SyntaxHighlighter
//                             style={dracula}
//                             PreTag="div"
//                             language={match[1]}
//                             {...props}
//                         >
//                             {String(codeChildren).replace(/\n$/, "")}
//                         </SyntaxHighlighter>
//                     ) : (
//                         <code className={className} {...props}>
//                             {codeChildren}
//                         </code>
//                     );
//                 },

//                 // p({ node, children, ...props }) {
//                 //     return (
//                 //         <p
//                 //             style={{ marginBottom: "0.5rem", whiteSpace: "pre-line" }}
//                 //             {...props}
//                 //         >
//                 //             {children}
//                 //         </p>
//                 //     );
//                 // },
//             }}
//         >
//             {children}
//         </Markdown>
//     );
// }

import React from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { dracula } from "react-syntax-highlighter/dist/cjs/styles/prism";

export default function MarkdownRenderer({ children }) {
    return (
        <Markdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeRaw]}
            components={{
                code({ inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || "");

                    if (!inline && match) {
                        return (
                            <SyntaxHighlighter
                                language={match[1]}
                                style={dracula}
                                PreTag="div"
                                customStyle={{
                                    margin: 0,
                                    padding: 0,
                                    background: "transparent",
                                }}
                                codeTagProps={{
                                    style: {
                                        padding: 0,
                                        margin: 0,
                                    },
                                }}
                                {...props}
                            >
                                {String(children).replace(/\n$/, "")}
                            </SyntaxHighlighter>
                        );
                    }

                    return (
                        <code
                            style={{
                                padding: 0,
                                margin: 0,
                                background: "transparent",
                            }}
                            className={className}
                            {...props}
                        >
                            {children}
                        </code>
                    );
                },
            }}
        >
            {children}
        </Markdown>
    );
}
