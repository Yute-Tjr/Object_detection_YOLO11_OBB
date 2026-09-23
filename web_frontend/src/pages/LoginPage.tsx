import { LockKey, Scan, User } from "@phosphor-icons/react";
import { type FormEvent, useState } from "react";


interface LoginPageProps {
  onLogin: (username: string, password: string) => Promise<void>;
}


export function LoginPage({ onLogin }: LoginPageProps) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await onLogin(username, password);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "登录失败，请稍后重试");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="login-page">
      <section className="login-card" aria-labelledby="login-title">
        <div className="login-card__brand" aria-hidden="true">
          <Scan size={30} weight="bold" />
        </div>
        <p className="login-card__eyebrow">生产线视觉质检</p>
        <h1 id="login-title">端子智能检测</h1>
        <p className="login-card__description">请使用已由管理员配置的账户登录系统</p>

        <form className="login-form" onSubmit={handleSubmit}>
          <label htmlFor="login-username">用户名</label>
          <div className="login-input">
            <User size={19} aria-hidden="true" />
            <input
              id="login-username"
              name="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              autoComplete="username"
              autoFocus
              required
            />
          </div>

          <label htmlFor="login-password">密码</label>
          <div className="login-input">
            <LockKey size={19} aria-hidden="true" />
            <input
              id="login-password"
              name="password"
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="current-password"
              required
            />
          </div>

          {error ? <p className="login-form__error" role="alert">{error}</p> : null}
          <button
            className="button button--primary login-form__submit"
            type="submit"
            disabled={submitting || !username || !password}
          >
            {submitting ? "正在登录…" : "登录"}
          </button>
        </form>
      </section>
    </main>
  );
}
