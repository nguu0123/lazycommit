package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"al.essio.dev/pkg/shellescape"
	"github.com/coder/pretty"
	"github.com/muesli/termenv"
	"github.com/sashabaranov/go-openai"
	"github.com/spf13/cobra"
)

var (
	colorProfile = termenv.ColorProfile()
	debugMode    = os.Getenv("LAZYCOMMIT_DEBUG") != ""
)

type runOptions struct {
	client        *openai.Client
	openAIBaseURL string
	model         string
	dryRun        bool
	amend         bool
	ref           string
	context       []string
}

func errorf(format string, args ...any) {
	c := pretty.FgColor(colorProfile.Color("#ff0000"))
	pretty.Fprintf(os.Stderr, c, "err: "+format, args...)
}

func debugf(format string, args ...any) {
	if !debugMode {
		return
	}
	c := pretty.FgColor(colorProfile.Color("#808080"))
	pretty.Fprintf(os.Stderr, c, "debug: "+format+"\n", args...)
}

func getLastCommitHash() (string, error) {
	cmd := exec.Command("git", "rev-parse", "HEAD")
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(output)), nil
}

func resolveRef(ref string) (string, error) {
	cmd := exec.Command("git", "rev-parse", ref)
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(output)), nil
}

func formatShellCommand(cmd *exec.Cmd) string {
	buf := &strings.Builder{}
	buf.WriteString(filepath.Base(cmd.Path))
	for _, arg := range cmd.Args[1:] {
		buf.WriteString(" ")
		buf.WriteString(shellescape.Quote(arg))
	}
	return buf.String()
}

func cleanAIMessage(msg string) string {
	// As reported by Ben, sometimes GPT returns the message
	// wrapped in a code block.
	if strings.HasPrefix(msg, "```") {
		msg = strings.TrimSuffix(msg, "```")
		msg = strings.TrimPrefix(msg, "```")
	}
	msg = strings.TrimSpace(msg)
	return msg
}

func run(opts runOptions) error {
	workdir, err := os.Getwd()
	if err != nil {
		return err
	}

	if opts.ref != "" && opts.amend {
		return errors.New("cannot use both [ref] and --amend")
	}

	var hash string
	if opts.amend {
		hash, err = getLastCommitHash()
		if err != nil {
			return err
		}
	} else if opts.ref != "" {
		hash, err = resolveRef(opts.ref)
		if err != nil {
			return fmt.Errorf("resolve ref %q: %w", opts.ref, err)
		}
	}

	msgs, err := BuildPrompt(os.Stdout, workdir, hash, opts.amend, 128000)
	if err != nil {
		return err
	}

	if len(opts.context) > 0 {
		msgs = append(msgs, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: "The user has provided additional context that MUST be included in the commit message",
		})
		for _, context := range opts.context {
			msgs = append(msgs, openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleUser,
				Content: context,
			})
		}
	}

	ctx := context.Background()
	stream, err := opts.client.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{
		Model:       opts.model,
		Stream:      true,
		Temperature: 0,
		StreamOptions: &openai.StreamOptions{
			IncludeUsage: true,
		},
		Messages: msgs,
	})
	if err != nil {
		return err
	}
	defer stream.Close()

	msg := &bytes.Buffer{}
	color := pretty.FgColor(colorProfile.Color("#2FA8FF"))

	for {
		resp, err := stream.Recv()
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
		// NOTE: don't know why but when len is 0, it's finished
		if len(resp.Choices) == 0 {
			break
		}
		c := resp.Choices[0].Delta.Content

		msg.WriteString(c)
		pretty.Fprintf(os.Stdout, color, "%s", c)
	}
	fmt.Println()

	cmd := exec.Command("git", "commit", "-m", msg.String())
	if opts.amend {
		cmd.Args = append(cmd.Args, "--amend")
	}

	if opts.dryRun {
		fmt.Println("Run the following command to commit:")
		fmt.Println(formatShellCommand(cmd))
		return nil
	}

	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stdout
	cmd.Stdin = os.Stdin
	return cmd.Run()
}

func main() {
	var opts runOptions
	var openAIKey string

	// Check if the key is empty

	rootCmd := &cobra.Command{
		Use:   "lazycommit [ref]",
		Short: "Commit message generator using LLM",
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) > 0 {
				opts.ref = args[0]
			}

			if openAIKey == "" {
				fmt.Fprintln(os.Stderr, "OPENAI_API_KEY is not set")
				os.Exit(1)
			}
			client := openai.NewClient(openAIKey)
			opts.client = client

			return run(opts)
		},
	}

	rootCmd.Flags().StringVarP(&opts.model, "model", "m", "gpt-4o-2024-08-06", "The model to use")
	rootCmd.Flags().StringVar(&openAIKey, "openai-key", os.Getenv("OPENAI_API_KEY"), "The OpenAI API key")
	rootCmd.Flags().StringVar(&opts.openAIBaseURL, "openai-base-url", "https://api.openai.com/v1", "The base URL for OpenAI API")
	rootCmd.Flags().BoolVarP(&opts.dryRun, "dry-run", "d", false, "Dry run the commit command")
	rootCmd.Flags().BoolVarP(&opts.amend, "amend", "a", false, "Amend the last commit")
	rootCmd.Flags().StringSliceVarP(&opts.context, "context", "c", nil, "Additional context for commit message")

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
