import {
  createContext,
  type ReactNode,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

import { ApiError, apiClient, subscribeUnauthorized } from "../api/client";
import type { AuthUser } from "../api/types";


export type AuthStatus = "loading" | "authenticated" | "anonymous";

export interface AuthClient {
  getCurrentUser(signal?: AbortSignal): Promise<AuthUser>;
  login(username: string, password: string, signal?: AbortSignal): Promise<AuthUser>;
  logout(signal?: AbortSignal): Promise<void>;
}

interface AuthContextValue {
  status: AuthStatus;
  user: AuthUser | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

interface AuthProviderProps {
  children: ReactNode;
  client?: AuthClient;
}

const AuthContext = createContext<AuthContextValue | null>(null);


export function AuthProvider({ children, client = apiClient }: AuthProviderProps) {
  const [status, setStatus] = useState<AuthStatus>("loading");
  const [user, setUser] = useState<AuthUser | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void client.getCurrentUser(controller.signal).then((currentUser) => {
      setUser(currentUser);
      setStatus("authenticated");
    }).catch((error: unknown) => {
      if (controller.signal.aborted) return;
      if (!(error instanceof ApiError) || error.status !== 401) {
        console.error("读取登录会话失败", error);
      }
      setUser(null);
      setStatus("anonymous");
    });
    return () => controller.abort();
  }, [client]);

  useEffect(() => subscribeUnauthorized(() => {
    setUser(null);
    setStatus("anonymous");
  }), []);

  const value = useMemo<AuthContextValue>(() => ({
    status,
    user,
    async login(username: string, password: string) {
      const currentUser = await client.login(username, password);
      setUser(currentUser);
      setStatus("authenticated");
    },
    async logout() {
      try {
        await client.logout();
      } finally {
        setUser(null);
        setStatus("anonymous");
      }
    },
  }), [client, status, user]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}


export function useAuth(): AuthContextValue {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth 必须在 AuthProvider 内使用");
  return value;
}
